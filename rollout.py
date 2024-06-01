import torch
import compiler_gym
from torch_geometric.data import Data, Batch
from torch.distributions import Categorical
import numpy as np
from collections import deque
import random
import pandas as pd
from reward import RuntimeReward

from graph_embedding import LlvmGraphRepresentationWithInst2vec

class ReplayBuffer:
    def __init__(self, data_names, buffer_size, mini_batch_size, device):
        self.data_keys = data_names
        self.data_dict = {}
        self.buffer_size = buffer_size
        self.mini_batch_size = mini_batch_size
        self.device = device

        self.reset()

    def reset(self):
        # Create a deque for each data type with set max length
        for name in self.data_keys:
                self.data_dict[name] = deque(maxlen=self.buffer_size)

    def buffer_full(self):
        return len(self.data_dict[self.data_keys[0]]) == self.buffer_size

    def data_log(self, data_name, data, detach=True, preprocess=None):
        # split tensor along batch into a list of individual datapoints
        # Extend the deque for data type, deque will handle popping old data to maintain buffer size
        if detach and data and "cpu" in dir(data[0]):
            data = [d.cpu().detach() for d in data]
        if preprocess is not None:
            data = [preprocess(d) for d in data]
        self.data_dict[data_name].extend(data)

    def __iter__(self):
        total_size = len(self.data_dict[self.data_keys[0]])

        if total_size <= self.mini_batch_size:
            mini_batch_size = total_size
        else:
            mini_batch_size = self.mini_batch_size
            total_size = total_size - total_size % self.mini_batch_size

        ids = np.random.permutation(total_size)
        ids = np.split(ids, total_size // mini_batch_size)
        for i in range(len(ids)):
                batch_dict = {}
                for name in self.data_keys:
                        c = [self.data_dict[name][j] for j in ids[i]]
                        if c and isinstance(c[0], Data):
                                batch_dict[name] = Batch.from_data_list(c)
                                batch_dict[name] = batch_dict[name].to(self.device)
                        elif c and isinstance(c[0], torch.Tensor):
                                if len(c[0].shape)==0:
                                        c = [x.reshape(1) for x in c]
                                batch_dict[name] = torch.cat(c).to(self.device)
                        else:
                                raise ValueError(f"cannot create batch for key '{name}'")
                batch_dict["batch_size"] = len(ids[i])
                yield batch_dict

    def __len__(self):
        return len(self.data_dict[self.data_keys[0]])

    def eval(self, keys):
        out = dict()
        for key in keys:
                out[f"avg_{key}"] = (sum(self.data_dict[key]) / len(self.data_dict[key])).mean().item() # type: ignore
        return out


def compute_gae(next_value, rewards, values, gamma=0.999, tau=0.95):
    gae = 0
    returns = deque()
    gae_logger = deque()

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value - values[step]
        gae = delta + gamma * tau * gae
        next_value = values[step]
        # If we add the value back to the GAE we get a TD approximation for the returns
        # which we can use to train the Value function
        returns.appendleft(gae + values[step])
        gae_logger.appendleft(gae)

    return returns, gae_logger


def init_data_buffer(device, num_trajs=10, num_steps=128, mini_batch_size=32, num_mini_batch=8):
    buffer_size = num_steps * num_trajs
    data_names = ["states", "next_states", "actions", "log_probs", "rewards"]
    return ReplayBuffer(data_names, buffer_size, mini_batch_size, device)

class MetricsLogger:
    def __init__(self, metric_names):
        self.dict = {metric: [] for metric in metric_names}

    def add(self, dict):
        assert set(dict.keys())==set(self.dict.keys())
        for key, value in dict.items():
            self.dict[key].append(value)
    
    def as_df(self):
        return pd.DataFrame(data=self.dict)


def rollout(model, benchmarks: list[str], data_buffer: ReplayBuffer, device, num_envs=10, num_trajs=1, num_steps=128, gamma=0.999, tau=0.95):
    benchmark_selection = benchmarks.copy()
    random.shuffle(benchmark_selection)
    benchmark_selection = benchmark_selection[:num_envs]

    metrics = MetricsLogger(("traj_len", "total_reward", "mean_reward", "ir_instr_count", "benchmark", "min_running_time"))

    model.eval()

    for benchmark in benchmark_selection:
        for _ in range(num_trajs):
            with torch.no_grad():
                with compiler_gym.make(
                    "llvm-v0",
                    benchmark=benchmark,
                ) as env:
                    env.reset()
                    env.runtime_observation_count = 25 # type: ignore

                    reward_eval = RuntimeReward(env)

                    states = []
                    next_states = []
                    actions = []
                    log_probs = []
                    # values = []
                    rewards = []

                    min_running_time = RuntimeReward.estimate_running_time(env)

                    current_state = LlvmGraphRepresentationWithInst2vec(env.observation).as_discrete_graph().to(device)

                    done = False
                    step = 0
                    while not done and step < num_steps:
                        action_probs = model(current_state)
                        action_dist = Categorical(probs=action_probs)
                        action = action_dist.sample()
                        observation, reward, done, info = env.step(action.item())

                        states.append(current_state)
                        actions.append(action)
                        log_probs.append(action_dist.log_prob(action))

                        if env.observation["Ir"] != current_state.ir:
                            current_state = LlvmGraphRepresentationWithInst2vec(env.observation).as_discrete_graph().to(device)
                        next_states.append(current_state)

                        running_time = RuntimeReward.estimate_running_time(env)
                        min_running_time = min(min_running_time, running_time)
                        rewards.append(reward_eval.rel_reward(env, running_time))

                        step += 1

                    data_buffer.data_log("states", states)
                    data_buffer.data_log("next_states", next_states)
                    data_buffer.data_log("actions", list(actions))
                    data_buffer.data_log("log_probs", list(log_probs))
                    data_buffer.data_log("rewards", list([torch.tensor(r).reshape(1).to(device) for r in rewards]))

                    metrics.add({
                          "traj_len": len(states),
                          "total_reward": sum(rewards),
                          "mean_reward": sum(rewards) / len(states),
                          "ir_instr_count": env.observation["IrInstructionCount"],
                          "benchmark": benchmark,
                          "min_running_time": min_running_time,
                    })

    model.train()

    return metrics
