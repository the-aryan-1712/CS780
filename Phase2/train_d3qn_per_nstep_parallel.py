from __future__ import annotations
import argparse, random
from collections import deque
from dataclasses import dataclass
from typing import Deque
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class DuelingDQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.value = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, n_actions)

    def forward(self, x):
        x = self.feature(x)
        v = self.value(x)
        a = self.advantage(x)
        return v + (a - a.mean(dim=1, keepdim=True))

@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool

class ReplayPER:
    def __init__(self, cap=100_000, alpha=0.6):
        self.cap = cap
        self.alpha = alpha
        self.pos = 0
        self.size = 0
        
        # Pre-allocate memory to avoid list appending overhead
        self.s_buf = np.zeros((cap, 18), dtype=np.float32)
        self.a_buf = np.zeros(cap, dtype=np.int64)
        self.r_buf = np.zeros(cap, dtype=np.float32)
        self.s2_buf = np.zeros((cap, 18), dtype=np.float32)
        self.d_buf = np.zeros(cap, dtype=np.float32)
        
        # SumTree for O(log N) sampling
        self.tree = np.zeros(2 * cap - 1)

    def add(self, t: Transition):
        max_prio = np.max(self.tree[-self.cap:]) if self.size > 0 else 1.0
        
        idx = self.pos
        self.s_buf[idx] = t.s
        self.a_buf[idx] = t.a
        self.r_buf[idx] = t.r
        self.s2_buf[idx] = t.s2
        self.d_buf[idx] = t.done
        
        self.update_tree(idx, max_prio ** self.alpha)
        
        self.pos = (self.pos + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def update_tree(self, idx, priority):
        tree_idx = idx + self.cap - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def sample(self, batch_size, beta=0.4):
        indices = []
        priorities = []
        segment = self.tree[0] / batch_size
        
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            v = random.uniform(a, b)
            
            # Efficient walk down the tree
            idx = 0
            while idx < self.cap - 1:
                left = 2 * idx + 1
                if v <= self.tree[left]:
                    idx = left
                else:
                    v -= self.tree[left]
                    idx = left + 1
            
            data_idx = idx - (self.cap - 1)
            indices.append(data_idx)
            priorities.append(self.tree[idx])

        # Importance sampling weights
        probs = np.array(priorities) / self.tree[0]
        weights = (self.size * probs) ** (-beta)
        weights /= weights.max()

        return (self.s_buf[indices], self.a_buf[indices], self.r_buf[indices], 
                self.s2_buf[indices], self.d_buf[indices], indices, weights)

    def update_priorities(self, indices, td_errors):
        priorities = (np.abs(td_errors) + 1e-5) ** self.alpha
        for idx, prio in zip(indices, priorities):
            self.update_tree(idx, prio)

    def __len__(self):
        return self.size


class NStepBuffer:
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.buffer = deque()

    def push(self, transition):
        self.buffer.append(transition)

    def get(self):
        if len(self.buffer) < self.n:
            return None

        R = 0
        for i in range(self.n):
            R += (self.gamma ** i) * self.buffer[i].r

        s = self.buffer[0].s
        a = self.buffer[0].a
        s_n = self.buffer[self.n - 1].s2
        done = self.buffer[self.n - 1].done

        self.buffer.popleft()

        return Transition(s, a, R, s_n, done)

    def reset(self):
        self.buffer.clear()


def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def soft_update(q, tgt, tau=0.005):
    for target_param, param in zip(tgt.parameters(), q.parameters()):
        target_param.data.copy_(
            tau * param.data + (1.0 - tau) * target_param.data
        )


class ParallelEnv:
    def __init__(self, OBELIX, num_envs, args):
        self.envs = [
            OBELIX(
                scaling_factor=args.scaling_factor,
                arena_size=args.arena_size,
                max_steps=args.max_steps,
                wall_obstacles=args.wall_obstacles,
                difficulty=args.difficulty,
                box_speed=args.box_speed,
                seed=args.seed + i,
            )
            for i in range(num_envs)
        ]
        self.num_envs = num_envs

    def reset(self, base_seed):
        states = []
        for i, env in enumerate(self.envs):
            s = env.reset(seed=base_seed + i)
            states.append(s)
        return np.array(states)

    def step(self, actions):
        next_states, rewards, dones = [], [], []

        for env, a in zip(self.envs, actions):
            s2, r, d = env.step(ACTIONS[a],render=False)
            next_states.append(s2)
            rewards.append(r)
            dones.append(d)

        return (
            np.array(next_states),
            np.array(rewards),
            np.array(dones),
        )

def process_rewards_vectorized(next_states, rewards):
    rewards[rewards == -1.0] = -0.2
    rewards += next_states[:, 17] * 180.0
    forward_far = np.sum(next_states[:, 4:12:2], axis=1)
    forward_near = np.sum(next_states[:, 5:12:2], axis=1)
    rewards += 1.0 * forward_far + 2.0 * forward_near
    rewards += next_states[:, 16] * 5.0
    side_signal = np.sum(next_states[:, 0:4], axis=1) + np.sum(next_states[:, 12:16], axis=1)
    rewards -= 0.5 * side_signal
    
    return rewards
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights.pth")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)

    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--replay", type=int, default=100000)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--target_sync", type=int, default=2000)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.1)
    ap.add_argument("--eps_decay_steps", type=int, default=20000000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_envs", type=int, default=4)
    args = ap.parse_args()

    beta = 0.4
    beta_increment = 5e-7

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    q = DuelingDQN().to(device)
    tgt = DuelingDQN().to(device)
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt = optim.Adam(q.parameters(), lr=args.lr)
    replay = ReplayPER(args.replay)
    steps = 0
    n_step = 2
    n_buffers = [NStepBuffer(n_step, args.gamma) for _ in range(args.num_envs)]
    total_returns = []

    def eps_by_step(t):
        if t >= args.eps_decay_steps:
            return args.eps_end
        frac = t / args.eps_decay_steps
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    progress = tqdm.trange(args.episodes)
    env = ParallelEnv(OBELIX, args.num_envs, args)
    states = env.reset(args.seed)

    for ep in progress:
        ep_ret = np.zeros(args.num_envs)

        for _ in range(args.max_steps):

            eps = eps_by_step(steps)

            if random.random() < eps:
                actions = np.random.randint(len(ACTIONS), size=args.num_envs)
            else:
                with torch.no_grad():
                    s_t = torch.from_numpy(states).float().to(device)
                    qs = q(s_t)
                    actions = torch.argmax(qs, dim=1).cpu().numpy()

            next_states, rewards, dones = env.step(actions)

            rewards = process_rewards_vectorized(next_states, rewards)

            for i in range(args.num_envs):
                transition = Transition(
                    s=states[i],
                    a=int(actions[i]),
                    r=float(rewards[i]),
                    s2=next_states[i],
                    done=bool(dones[i]),
                )

                n_buffers[i].push(transition)
                n_step_transition = n_buffers[i].get()

                if n_step_transition:
                    replay.add(n_step_transition)

                if dones[i]:
                    n_buffers[i].reset()

            for i in range(args.num_envs):
                if dones[i]:
                    next_states[i] = env.envs[i].reset(seed=args.seed + ep + i)
                    n_buffers[i].reset()

            states = next_states

            steps += args.num_envs
            beta = min(1.0, beta + beta_increment)

            if len(replay) >= max(args.warmup, args.batch) and (steps % (8 * args.num_envs) == 0):

                sb, ab, rb, s2b, db, idx, weights = replay.sample(args.batch, beta)

                sb_t = torch.from_numpy(sb).to(device)
                rb_t = torch.from_numpy(rb).to(device)
                ab_t = torch.from_numpy(ab).long().to(device)
                s2b_t = torch.from_numpy(s2b).to(device)
                db_t = torch.from_numpy(db).float().to(device)
                w_t = torch.from_numpy(weights).to(device)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    next_q = q(s2b_t)
                    next_a = torch.argmax(next_q, dim=1)
                    with torch.no_grad():
                        

                        next_q_tgt = tgt(s2b_t)
                        next_val = next_q_tgt.gather(1, next_a.unsqueeze(1)).squeeze(1)

                        gamma_n = args.gamma ** n_step
                        y = rb_t + gamma_n * (1.0 - db_t) * next_val

                    pred = q(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)

                    td_error = y - pred
                    loss = (w_t * td_error.pow(2)).mean()

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()

                replay.update_priorities(idx, td_error.detach().cpu().numpy())
                soft_update(q, tgt)

        progress.set_description(f"Eps: {eps:.3f} Beta: {beta:.3f}")
        total_returns.append(ep_ret.mean())  # FIXED

    torch.save(q.state_dict(), args.out)
    np.save("returns.npy", np.array(total_returns))
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
