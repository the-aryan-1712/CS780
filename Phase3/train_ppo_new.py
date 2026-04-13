from __future__ import annotations
import argparse, random, time, importlib.util
from collections import deque
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import multiprocessing as mp

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class PPOAgent(nn.Module):
    def __init__(self, in_dim=72, n_actions=5):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        features = self.base(x)
        return self.actor(features), self.critic(features)

    @torch.no_grad()
    def get_action(self, x):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

def env_worker(remote, obelix_py, env_kwargs):
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    env = mod.OBELIX(**env_kwargs)
    
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            s, r, d = env.step(data, render=False)
            if d: s = env.reset()
            remote.send((s, r, d))
        elif cmd == 'reset':
            s = env.reset(seed=data)
            remote.send(s)
        elif cmd == 'close':
            remote.close()
            break

class VecEnv:
    def __init__(self, num_envs, obelix_py, env_kwargs):
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.ps = [mp.Process(target=env_worker, args=(work, obelix_py, env_kwargs))
                   for work in self.work_remotes]
        for p in self.ps: p.start()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones)

    def reset(self, seeds):
        for remote, seed in zip(self.remotes, seeds):
            remote.send(('reset', seed))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        for remote in self.remotes: remote.send(('close', None))
        for p in self.ps: p.join()

def main():
    # Use 'spawn' for macOS compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--num_envs", type=int, default=mp.cpu_count() - 2) # Leave 2 cores for OS to reduce heat
    ap.add_argument("--episodes", type=int, default=10000)
    ap.add_argument("--steps_per_batch", type=int, default=2048)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--n_epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--stack_size", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on {device} using {args.num_envs} CPU workers...")

    env_kwargs = {'scaling_factor': 5, 'arena_size': 500, 'max_steps': 5000, 'difficulty': 3}
    envs = VecEnv(args.num_envs, args.obelix_py, env_kwargs)
    
    agent = PPOAgent(in_dim=18 * args.stack_size).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr)

    obs = envs.reset([random.randint(0, 1000) for _ in range(args.num_envs)])
    state_stacks = [deque([obs[i]] * args.stack_size, maxlen=args.stack_size) for i in range(args.num_envs)]
    
    # Track metrics
    episode_rewards = deque(maxlen=20) 
    current_rewards = np.zeros(args.num_envs)

    progress = tqdm.trange(args.episodes)
    for i in progress:
        mb_obs, mb_actions, mb_probs, mb_values, mb_rewards, mb_dones = [], [], [], [], [], []
        
        # 1. Experience Collection
        for _ in range(args.steps_per_batch // args.num_envs):
            current_states = np.stack([np.concatenate(list(state_stacks[j])) for j in range(args.num_envs)])
            states_t = torch.as_tensor(current_states, dtype=torch.float32, device=device)
            
            actions_t, probs_t, values_t = agent.get_action(states_t)
            
            actions = [ACTIONS[a] for a in actions_t.cpu().numpy()]
            next_obs, rewards, dones = envs.step(actions)
            
            mb_obs.append(current_states)
            mb_actions.append(actions_t.cpu().numpy())
            mb_probs.append(probs_t.cpu().numpy())
            mb_values.append(values_t.cpu().squeeze().numpy())
            mb_rewards.append(rewards)
            mb_dones.append(dones)

            current_rewards += rewards
            for idx, done in enumerate(dones):
                if done:
                    episode_rewards.append(current_rewards[idx])
                    current_rewards[idx] = 0

            for j in range(args.num_envs):
                state_stacks[j].append(next_obs[j])

        # Update Progress Bar with Average Reward
        avg_rew = np.mean(episode_rewards) if len(episode_rewards) > 0 else 0
        progress.set_description(f"Avg Reward: {avg_rew:.2f}")

        # 2. Compute GAE and Returns
        mb_obs = np.concatenate(mb_obs); mb_actions = np.concatenate(mb_actions)
        mb_probs = np.concatenate(mb_probs); mb_values = np.concatenate(mb_values)
        mb_rewards = np.concatenate(mb_rewards); mb_dones = np.concatenate(mb_dones)

        advantages = np.zeros_like(mb_rewards)
        last_gae = 0
        for t in reversed(range(len(mb_rewards))):
            if t == len(mb_rewards) - 1: next_non_terminal = 0; next_values = 0
            else: next_non_terminal = 1.0 - mb_dones[t]; next_values = mb_values[t+1]
            delta = mb_rewards[t] + args.gamma * next_values * next_non_terminal - mb_values[t]
            advantages[t] = last_gae = delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae
        returns = advantages + mb_values

        # 3. Update Model
        b_obs = torch.as_tensor(mb_obs, dtype=torch.float32, device=device)
        b_actions = torch.as_tensor(mb_actions, device=device)
        b_probs = torch.as_tensor(mb_probs, device=device)
        b_advantages = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        b_returns = torch.as_tensor(returns, dtype=torch.float32, device=device)

        for _ in range(args.n_epochs):
            new_logits, new_values = agent(b_obs)
            new_dist = Categorical(logits=new_logits)
            new_probs = new_dist.log_prob(b_actions)
            
            ratio = (new_probs - b_probs).exp()
            surr1 = ratio * b_advantages
            surr2 = torch.clamp(ratio, 1-0.2, 1+0.2) * b_advantages
            loss = -torch.min(surr1, surr2).mean() + 0.5 * (b_returns - new_values.squeeze()).pow(2).mean()
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    envs.close()
    torch.save(agent.state_dict(), "ppo_mac_final.pth")
    print("Training Complete! Model saved as ppo_mac_final.pth")

if __name__ == "__main__":
    main()
