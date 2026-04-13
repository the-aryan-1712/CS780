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
    def __init__(self, cap=100_000, alpha=0.4):
        self.cap = cap
        self.alpha = alpha
        self.buf = []
        self.priorities = np.zeros((cap,), dtype=np.float32)
        self.pos = 0

    def add(self, t):
        max_prio = self.priorities.max() if self.buf else 1.0

        if len(self.buf) < self.cap:
            self.buf.append(t)
        else:
            self.buf[self.pos] = t

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.cap

    def sample(self, batch, beta=0.4):
        prios = self.priorities[:len(self.buf)]
        probs = prios ** self.alpha
        if probs.sum() == 0:
            probs = np.ones_like(probs)
        probs /= probs.sum()

        idx = np.random.choice(len(self.buf), batch, p=probs)
        items = [self.buf[i] for i in idx]

        s = np.stack([it.s for it in items]).astype(np.float32)
        a = np.array([it.a for it in items])
        r = np.array([it.r for it in items])
        s2 = np.stack([it.s2 for it in items]).astype(np.float32)
        d = np.array([it.done for it in items])

        weights = (len(self.buf) * probs[idx]) ** (-beta)
        weights /= weights.max()

        return s, a, r, s2, d, idx, weights

    def update_priorities(self, idx, td_errors):
        for i, td in zip(idx, td_errors):
            self.priorities[i] = abs(td) + 1e-5

    def __len__(self):
        return len(self.buf)




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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, required=True)
    ap.add_argument("--out", type=str, default="weights.pth")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--max_steps", type=int, default=1000)
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
    ap.add_argument("--eps_decay_steps", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    beta = 0.4
    beta_increment = 5e-7

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    q = DuelingDQN()
    tgt = DuelingDQN()
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt = optim.Adam(q.parameters(), lr=args.lr)
    replay = ReplayPER(args.replay)
    steps = 0
    n_step = 2
    n_buffer = NStepBuffer(n_step, args.gamma)
    total_returns = []
    def eps_by_step(t):
        if t >= args.eps_decay_steps:
            return args.eps_end
        frac = t / args.eps_decay_steps
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    progress = tqdm.trange(args.episodes)
    for ep in progress:
        env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=args.seed + ep,
        )
        s = env.reset(seed=args.seed + ep)
        ep_ret = 0.0

        for _ in range(args.max_steps):
            eps = eps_by_step(steps)
            if np.random.rand() < eps:
                a = np.random.randint(len(ACTIONS))
            else:
                with torch.no_grad():
                    qs = q(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                a = int(np.argmax(qs))

            s2, r, done = env.step(ACTIONS[a], render=False)
            
            ep_ret += float(r)
            r = np.clip(r, -200, 200)
            forward_far = np.sum(s2[4:12:2])   # far sensors
            forward_near = np.sum(s2[5:12:2])  # near sensors
            r += 0.5 * forward_far
            r += 1.0 * forward_near
            if s2[16]:
                r += 3.0
            side_signal = np.sum(s2[0:4]) + np.sum(s2[12:16])
            r -= 0.2 * side_signal
            if s2[17]:
                r -= 5.0
            
            transition = Transition(s=s, a=a, r=float(r), s2=s2, done=bool(done))
            n_buffer.push(transition)

            n_step_transition = n_buffer.get()
            if n_step_transition:
                replay.add(n_step_transition)
            s = s2
            steps += 1
            beta = min(1.0, beta + beta_increment)
            if len(replay) >= max(args.warmup, args.batch) and (steps % 4 == 0):
                
                sb, ab, rb, s2b, db, idx, weights = replay.sample(args.batch, beta)
                sb_t = torch.tensor(sb, dtype=torch.float32)
                rb_t = torch.tensor(rb, dtype=torch.float32)
                ab_t = torch.tensor(ab, dtype=torch.long)
                s2b_t = torch.tensor(s2b, dtype=torch.float32)
                db_t = torch.tensor(db, dtype=torch.float32)
                w_t = torch.tensor(weights, dtype=torch.float32)

                with torch.no_grad():
                    next_q = q(s2b_t)
                    next_a = torch.argmax(next_q, dim=1)

                    next_q_tgt = tgt(s2b_t)
                    next_val = next_q_tgt.gather(1, next_a.unsqueeze(1)).squeeze(1)

                    gamma_n = args.gamma ** n_step
                    y = rb_t + gamma_n * (1.0 - db_t) * next_val

                pred = q(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)

                td_error = y - pred

                loss = (w_t * td_error.pow(2)).mean()
                # td_error = torch.clamp(td_error, -10, 10)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

                # update priorities
                replay.update_priorities(idx, td_error.detach().numpy())

                soft_update(q, tgt, tau=0.005)

            if done:
                break
            if ep % 100 == 0 and _ < 20:
                print("Forward:", np.sum(s2[4:12]), "IR:", s2[16])

        
        while True:
            n_step_transition = n_buffer.get()
            if not n_step_transition:
                break
            replay.add(n_step_transition)

        n_buffer.reset()
        total_returns.append(ep_ret)
        progress.set_description(f"Return: {ep_ret:.1f} Eps: {eps_by_step(steps):.3f} Beta: {beta:.3f}")

    torch.save(q.state_dict(), args.out)
    np.save("returns.npy", np.array(total_returns))
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
