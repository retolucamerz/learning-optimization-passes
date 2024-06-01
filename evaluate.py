import compiler_gym
from torch.distributions import Categorical
import pandas as pd

from graph_embedding import LlvmGraphRepresentationWithInst2vec
from reward import RuntimeReward

def policy_from_model(model):
    def policy(state_graph, *args, **kwargs):
        action_probs = model(state_graph)
        action_dist = Categorical(probs=action_probs)
        return action_dist.sample().item()

    return policy

def evaluate(benchmarks, policy, device, num_steps, num_trajs_per_benchmark=1, eval_runtime_per_step=False):
    data = {
        "step": [],
        "actions": [],
        "rewards": [],
        "ir_instr_count": [],
        "repeat": [],
        "benchmark": [],
    }
    if eval_runtime_per_step:
        data["running_time"] = []

    final_data = {
        "num_steps": [],
        "ir": [],
        "running_time": [],
        "ir_instr_count": [],
        "repeat": [],
        "benchmark": [],
    }
    if eval_runtime_per_step:
        final_data["min_running_time"] = []

    for benchmark in benchmarks:
        for repeat in range(num_trajs_per_benchmark):
            with compiler_gym.make(
                    "llvm-v0",
                    benchmark=benchmark,
                ) as env:
                    env.reset()
                    env.runtime_observation_count = 25 # type: ignore
                    reward_eval = RuntimeReward(env)

                    if eval_runtime_per_step:
                        min_running_time = RuntimeReward.estimate_running_time(env)

                    done = False
                    step = 0
                    while not done and step < num_steps:
                        state_graph = LlvmGraphRepresentationWithInst2vec(env.observation).as_discrete_graph().to(device)
                        action = policy(state_graph, env=env)
                        _, _, done, _ = env.step(action)

                        data["step"].append(step)
                        data["actions"].append(action)
                        if eval_runtime_per_step:
                            running_time = RuntimeReward.estimate_running_time(env)
                            data["running_time"].append(running_time)
                            data["rewards"].append(reward_eval.rel_reward(env, running_time=running_time))
                            min_running_time = min(min_running_time, running_time) # type: ignore
                        data["ir_instr_count"].append(env.observation["IrInstructionCount"])
                        data["repeat"].append(repeat)
                        data["benchmark"].append(benchmark)

                        step += 1

                    final_data["num_steps"].append(step)
                    final_data["ir"].append(env.observation["Ir"])
                    final_data["running_time"].append(RuntimeReward.estimate_running_time(env))
                    final_data["ir_instr_count"].append(env.observation["IrInstructionCount"])
                    final_data["repeat"].append(repeat)
                    final_data["benchmark"].append(benchmark)
                    if eval_runtime_per_step:
                        final_data["min_running_time"].append(min_running_time) # type: ignore

    if not eval_runtime_per_step:
        del data["running_time"]
        del data["rewards"]

    return pd.DataFrame(data=data), pd.DataFrame(data=final_data)
                    
