import torch
from torch.distributions import Categorical
from torch import optim
from torch import nn
from tqdm import tqdm
from torch_geometric.nn.conv import GINEConv, GATConv
import time
import wandb
import os.path as osp
import os
from policy import Model
from rollout import init_data_buffer, rollout
import pandas as pd

EPS = 1e-7

def ppo_loss(new_log_probs, old_log_probs, advantages, clip_ratio):
    ratio = (new_log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    actor_loss = torch.min(surr1, surr2)
    return actor_loss

def clipped_critic_loss(new_value, old_value, returns, clip_difference):
    old_value = old_value.detach()
    vf_loss1 = (new_value - returns).pow(2.)
    vpredclipped = old_value + torch.clamp(new_value - old_value, -clip_difference, clip_difference)
    vf_loss2 = (vpredclipped - returns).pow(2.)
    critic_loss = torch.max(vf_loss1, vf_loss2)
    return critic_loss

def ppo_update(model, optimizer, data_buffer, ppo_epochs, clip_policy_ratio, clip_critic_diff, clip_grad_norm=40, normailze_advantage=True, entropy_factor=0.001):
    for e in tqdm(range(ppo_epochs)):
        for data_batch in data_buffer:
            new_probs, new_value = model(data_batch["states"])
            new_dist = Categorical(new_probs)
            new_log_probs = new_dist.log_prob(data_batch["actions"])

            advantages = data_batch["advantages"]
            if normailze_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)
            actor_loss = ppo_loss(new_log_probs, data_batch["log_probs"], advantages, clip_policy_ratio).mean()
            critic_loss = clipped_critic_loss(new_value, data_batch["values"], data_batch["returns"], clip_critic_diff).mean()
            entropy = new_dist.entropy().mean()
            loss = critic_loss - actor_loss - entropy_factor * entropy

            optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm to further prevent large updates
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

if __name__=="__main__":
    lr = 5e-4
    num_mini_batch = 8 # How many minibatchs (therefore optimization steps) we want per epoch 
    num_steps = 50 # Total number of steps during the rollout phase 
    ppo_epochs = 5 # Number of epochs for training
    gamma = 0.98
    tau = 0.95
    clip_policy_ratio = 0.1
    clip_critic_diff = 0.2
    entropy_factor = 0.001
    num_trajs_per_prgm = 5
    max_iter = 500
    conv = GINEConv # GATConv
    normailze_advantage = False

    use_wandb = False
    save_model = False

    device = torch.device(0 if torch.cuda.is_available() else "cpu")
    
    benchmarks = [
        "benchmark://cbench-v1/qsort",
    ]

    model_id = f"{conv.__name__}_{num_steps}_{time.strftime('%Y%m%d-%H%M%S')}"
    if use_wandb:
        run = wandb.init(
            project="rl-optimization-passes",
            name=model_id,
            config={
                "learning_rate": lr,
                "num_mini_batch": num_mini_batch,
                "num_steps": num_steps,
                "ppo_epochs": ppo_epochs,
                "gamma": gamma,
                "tau": tau,
                "clip_policy_ratio": clip_policy_ratio,
                "clip_critic_diff": clip_critic_diff,
                "num_benchmarks": len(benchmarks),
                "entropy_factor": entropy_factor,
                "conv": conv.__name__,
                "algo": "ppo",
                "benchmarks": str(benchmarks),
                "normailze_advantage": normailze_advantage,
                "num_trajs_per_prgm": num_trajs_per_prgm,
            },
        )

    model_dir = osp.join("models", model_id)
    os.mkdir(model_dir)

    ACTION_SPACE = 124
    model = Model(num_outputs=ACTION_SPACE, conv=conv).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    data_buffer = init_data_buffer(device, num_trajs=10, num_steps=num_steps, num_mini_batch=num_mini_batch)
    metrics = []

    iter = 0
    while iter < max_iter:
        # rollout and eval
        iter_metrics = rollout(model, benchmarks, data_buffer, device, num_steps=num_steps, num_envs=1, num_trajs=num_trajs_per_prgm, gamma=gamma, tau=tau)

        iter_df = iter_metrics.as_df()
        if use_wandb:
            eval_dict = dict(iter_df.select_dtypes(include='number').mean())
            # eval_dict["min_running_time"] = iter_df.groupby()
            run.log(eval_dict) # type: ignore
        iter_df["iteration"] = iter
        iter_df["model"] = model_id
        metrics.append(iter_df)
        pd.concat(metrics).to_csv(osp.join(model_dir, "metrics.csv"))

        # policy update
        ppo_update(model, optimizer, data_buffer, ppo_epochs, clip_policy_ratio, clip_critic_diff, clip_grad_norm=40, entropy_factor=entropy_factor, normailze_advantage=normailze_advantage)

        iter += 1
        if save_model:
            torch.save(model.state_dict(), osp.join(model_dir, f"{model_id}-XXX.pt"))
            if iter%10==0:
                torch.save(model.state_dict(), osp.join(model_dir, f"{model_id}-{iter:03d}.pt"))
