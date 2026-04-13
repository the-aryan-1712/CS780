import os
import numpy as np
import torch
import torch.nn as nn
from collections import deque

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

# Global persistent variables
_MODEL = None
_STACK = None

class PPOAgent(nn.Module):
    """Architecture must exactly match the training script."""
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
        return self.actor(features)

def _load_once():
    """Load weights and initialize the history buffer."""
    global _MODEL, _STACK
    if _MODEL is not None:
        return

    # Load weights from the current directory
    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "weights.pth")

    model = PPOAgent(in_dim=72, n_actions=5)
    
    # map_location="cpu" ensures it works even if you trained on MPS/CUDA
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    model.eval()
    _MODEL = model

    # Initialize the deque with zeros to handle the first few steps
    _STACK = deque([np.zeros(18, dtype=np.float32)] * 4, maxlen=4)

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """Processes observation through the frame stack and returns an action."""
    global _STACK
    _load_once()

    # 1. Update the rolling window of frames
    _STACK.append(obs.astype(np.float32))
    
    # 2. Concatenate the 4 frames into a 72-dim vector
    stacked_input = np.concatenate(list(_STACK))
    
    # 3. Convert to torch tensor and ensure it is Float32
    x = torch.from_numpy(stacked_input).float().unsqueeze(0)

    # 4. Predict
    with torch.no_grad():
        logits = _MODEL(x).squeeze(0).numpy()

    # 5. Greedy action selection
    action_idx = int(np.argmax(logits))
    return ACTIONS[action_idx]
