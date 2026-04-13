"""
Submission template (USES trained weights).

Use this template if your agent depends on a trained neural network.
Place your saved model file (weights.pth) inside the submission folder.

The policy loads the model and uses it to predict the best action
from the observation.

The evaluator will import this file and call `policy(obs, rng)`.
"""

import os
import numpy as np

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None  # stores the loaded model


def _load_once():
    """Load the trained model and weights."""
    global _MODEL
    if _MODEL is not None:
        return

    submission_dir = os.path.dirname(__file__)
    wpath = os.path.join(submission_dir, "weights.pth")

    import torch
    import torch.nn as nn

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

    model = DuelingDQN()
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    model.eval()

    _MODEL = model


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()

    import torch
    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        logits = _MODEL(x).squeeze(0).numpy()

    return ACTIONS[int(np.argmax(logits))]
