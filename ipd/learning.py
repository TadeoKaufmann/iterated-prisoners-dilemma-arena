"""Independent Q-learners that discover how to play against each other.

State: the previous round from the learner's side (CC, CD, DC, DD, or START).
Actions: 0 = cooperate, 1 = defect.
"""

import numpy as np

from .game import AXELROD, SWAP, noisy
from .strategies import Strategy

START = 4


def as_strategy(q, name="learned"):
    """Greedy policy of a Q table as a memory-one strategy."""
    greedy = (q[:, 0] > q[:, 1]).astype(float)
    return Strategy(name, float(greedy[START]), tuple(float(v) for v in greedy[:4]))


def train(opponent=None, episodes=300_000, alpha=0.05, gamma=0.95,
          explore=(1.0, 0.001), restart=0.01, payoffs=AXELROD, seed=0):
    """Train a learner against `opponent`, a Strategy, or against a second
    learner when it is None. Exploration decays geometrically and the match
    restarts with probability `restart`. Returns the greedy strategies."""
    rng = np.random.default_rng(seed)
    q = [[[0.0, 0.0] for _ in range(5)] for _ in range(2)]
    reward = payoffs.vector.tolist()
    decay = (explore[1] / explore[0]) ** (1 / episodes)
    eps = explore[0]
    draws = rng.random((episodes, 7))
    state = START

    for u in draws.tolist():
        views = (state, START if state == START else SWAP[state])
        actions = []
        for k in range(2):
            if k == 1 and opponent is not None:
                p = opponent.first if views[1] == START else opponent.p[views[1]]
                actions.append(0 if u[1] < p else 1)
            elif u[2 + k] < eps:
                actions.append(int(u[4 + k] < 0.5))
            else:
                c, d = q[k][views[k]]
                actions.append(0 if c > d else 1 if d > c else int(u[4 + k] < 0.5))
        nxt = 2 * actions[0] + actions[1]
        next_views = (nxt, SWAP[nxt])
        for k in range(1 if opponent is not None else 2):
            row = q[k][views[k]]
            target = reward[next_views[k]] + gamma * max(q[k][next_views[k]])
            row[actions[k]] += alpha * (target - row[actions[k]])
        state = START if u[6] < restart else nxt
        eps *= decay

    first = as_strategy(np.array(q[0]))
    return first if opponent is not None else (first, as_strategy(np.array(q[1])))
