from policy import Model
from torch_geometric.nn.conv import GINEConv
from tqdm import tqdm
from torch.distributions import Categorical
import torch.nn.functional as F
from torch import optim
import torch
from itertools import chain
import time
import wandb
import os.path as osp
import os
from rollout import init_data_buffer, rollout
import pandas as pd

def update_q(qf1, qf2, q_optim, data_batch, gamma=0.99):
    with torch.no_grad():
        next_pi = policy(data_batch["next_states"])
        next_q_sampled = torch.min(
            (next_pi * qf1(data_batch["next_states"])).sum(dim=-1),
            (next_pi * qf2(data_batch["next_states"])).sum(dim=-1),
        )
        target_q = 500*data_batch["rewards"] + gamma * next_q_sampled
    batch_size = next_pi.shape[0]
    actions = data_batch["actions"]
    q1 = qf1(data_batch["states"])[range(batch_size), actions]
    q2 = qf2(data_batch["states"])[range(batch_size), actions]
    qf1_loss = F.mse_loss(q1, target_q)
    qf2_loss = F.mse_loss(q2, target_q)
    qf_loss = qf1_loss + qf2_loss

    q_optim.zero_grad()
    qf_loss.backward()
    q_optim.step()
    return qf_loss.item()

def update_policy(policy, qf1, qf2, policy_optim, data_batch, entropy_factor=0.01):
    pi = policy(data_batch["states"])
    with torch.no_grad():
        q1 = qf1(data_batch["states"])
        q2 = qf2(data_batch["states"])
    q_sampled = torch.min(
        (pi * q1).sum(dim=-1),
        (pi * q2).sum(dim=-1),
    )
    policy_q_loss = -q_sampled.mean()

    dist = Categorical(pi)
    entropy_loss = -dist.entropy().mean()

    policy_loss = policy_q_loss + entropy_factor * entropy_loss
    policy_optim.zero_grad()
    policy_loss.backward()
    policy_optim.step()
    return policy_q_loss.item(), entropy_loss.item()

def sac_update(policy, qf1, qf2, policy_optim, q_optim, data_buffer, epochs, entropy_factor=0.001, gamma=0.99, run=None, base_epoch=0):
    print("started SAC update")
    q_losses = []
    policy_losses = []
    entropies = []

    for e in range(epochs):
        epoch_q_losses = []
        epoch_policy_losses = []
        for data_batch in data_buffer:
            # Q update
            qf_loss = update_q(qf1, qf2, q_optim, data_batch, gamma=gamma)
            epoch_q_losses.append(qf_loss)

            # policy update
            policy_q_loss, entropy_loss = update_policy(policy, qf1, qf2, policy_optim, data_batch, entropy_factor=entropy_factor)
            epoch_policy_losses.append(policy_q_loss)
            entropies.append(entropy_loss)

        mean_q_loss = sum(epoch_q_losses) / len(epoch_q_losses)
        mean_policy_loss = sum(epoch_policy_losses) / len(epoch_policy_losses)
        print(f"epoch {e+1:02d}: {mean_q_loss:0.5f} (q), {mean_policy_loss:0.5f} (pol)")
        if run is not None:
            run.log({
                "epoch": e + base_epoch,
                "q_loss": mean_q_loss,
                "policy_loss": mean_policy_loss,
            })

        q_losses.extend(epoch_q_losses)
        policy_losses.extend(epoch_policy_losses)

    return q_losses, policy_losses, entropies


def learn_q(qf1, qf2, q_optim, data_buffer, epochs, gamma=0.99, base_epoch=0, run=None):
    print("started Q learning")
    q_losses = []

    for e in range(epochs):
        epoch_q_losses = []
        for data_batch in data_buffer:
            qf_loss = update_q(qf1, qf2, q_optim, data_batch, gamma=gamma)
            epoch_q_losses.append(qf_loss)
        mean_q_loss = sum(epoch_q_losses) / len(epoch_q_losses)
        print(f"epoch {e+1:02d}: {mean_q_loss:0.5f}")
        if run is not None:
            run.log({
                "epoch": e + base_epoch,
                "q_loss": mean_q_loss,
                "policy_loss": 0,
            })
        q_losses.extend(epoch_q_losses)

    return q_losses


if __name__=="__main__":
    device = torch.device(0 if torch.cuda.is_available() else "cpu")

    ACTION_SPACE = 124
    autophase_emb_dim = 50
    hidden_units = 100
    num_layers = 6
    conv = GINEConv

    entropy_factor = 0.01

    gamma = 0.99
    lr = 5e-4

    max_iter = 500
    num_trajs_per_prgm = 5
    num_trajs_in_buffer = 4*num_trajs_per_prgm
    num_steps = 50
    mini_batch_size = 32
    epochs = 15
    use_wandb = True
    save_model = True

    qf1 = Model(num_outputs=ACTION_SPACE, autophase_emb_dim=autophase_emb_dim, hidden_units=hidden_units, num_layers=5, conv=conv, use_softmax=False).to(device)
    qf2 = Model(num_outputs=ACTION_SPACE, autophase_emb_dim=autophase_emb_dim, hidden_units=hidden_units, num_layers=5, conv=conv, use_softmax=False).to(device)
    q_optim = optim.Adam(chain(qf1.parameters(), qf2.parameters()), lr=lr)

    policy = Model(num_outputs=ACTION_SPACE, autophase_emb_dim=autophase_emb_dim, hidden_units=hidden_units, num_layers=num_layers, conv=conv, use_softmax=True).to(device)
    policy_optim = optim.Adam(policy.parameters(), lr=lr)

    benchmarks = [
        "benchmark://cbench-v1/qsort",
        # "benchmark://cbench-v1/sha",
        # "benchmark://cbench-v1/stringsearch",
    ]

    model_id = f"SAC_{conv.__name__}_{num_steps}_{time.strftime('%Y%m%d-%H%M%S')}"
    model_dir = osp.join("models", model_id)
    os.mkdir(model_dir)
    if use_wandb:
        run = wandb.init(
            project="rl-optimization-passes",
            name=model_id,
            config={
                "learning_rate": lr,
                "mini_batch_size": mini_batch_size,
                "num_steps": num_steps,
                "epochs": epochs,
                "gamma": gamma,
                "num_benchmarks": len(benchmarks),
                "entropy_factor": entropy_factor,
                "conv": conv.__name__,
                "algo": "ppo",
                "benchmarks": str(benchmarks),
                "num_trajs_per_prgm": num_trajs_per_prgm,
                "num_trajs_in_buffer": num_trajs_in_buffer,
            },
        )
    else:
        run = None

    data_buffer = init_data_buffer(device, num_trajs=num_trajs_in_buffer, num_steps=num_steps, mini_batch_size=mini_batch_size)
    iter_metrics = rollout(policy, benchmarks, data_buffer, device, num_steps=num_steps, num_envs=1, num_trajs=num_trajs_in_buffer-num_trajs_per_prgm, gamma=gamma)

    metrics = []
    iter = 0
    while iter < max_iter:
        iter_metrics = rollout(policy, benchmarks, data_buffer, device, num_steps=num_steps, num_envs=1, num_trajs=num_trajs_per_prgm, gamma=gamma)

        iter_df = iter_metrics.as_df()
        eval_dict = dict(iter_df.select_dtypes(include='number').mean())
        print(f"iteration {iter:02d}: {eval_dict}")
        if use_wandb:
            eval_dict["iter"] = iter # type: ignore
            run.log(eval_dict) # type: ignore
        iter_df["iteration"] = iter
        iter_df["model"] = model_id
        metrics.append(iter_df)
        pd.concat(metrics).to_csv(osp.join(model_dir, "metrics.csv"))

        sac_update(policy, qf1, qf2, policy_optim, q_optim, data_buffer, epochs=epochs, entropy_factor=entropy_factor, gamma=gamma, run=run, base_epoch=q_learning_epochs+epochs*iter) # type: ignore

        iter += 1
        if save_model:
            torch.save(policy.state_dict(), osp.join(model_dir, f"policy-{model_id}-{iter:03d}.pt"))
            torch.save(qf1.state_dict(), osp.join(model_dir, f"qf1-{model_id}-{iter:03d}.pt"))
            torch.save(qf2.state_dict(), osp.join(model_dir, f"qf2-{model_id}-{iter:03d}.pt"))
