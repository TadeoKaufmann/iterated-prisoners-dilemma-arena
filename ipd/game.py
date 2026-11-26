"""Exact and simulated play between two memory-one strategies.

With memory-one players the previous round is a four-state Markov chain, so
expected payoffs can be computed exactly, noise included, with no sampling error.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Payoffs:
    T: float = 5.0  # temptation
    R: float = 3.0  # reward
    P: float = 1.0  # punishment
    S: float = 0.0  # sucker

    def __post_init__(self):
        if not (self.T > self.R > self.P > self.S and 2 * self.R > self.T + self.S):
            raise ValueError("not an iterated prisoner's dilemma")

    @property
    def vector(self):
        """My payoff in the states CC, CD, DC, DD."""
        return np.array([self.R, self.S, self.T, self.P])


AXELROD = Payoffs()
# The matrix used in the Game Theory course exercises at UBA.
COURSE = Payoffs(T=2.0, R=1.0, P=0.0, S=-1.0)

# The same round seen by the other player: CD and DC swap places.
SWAP = [0, 2, 1, 3]


def noisy(prob, noise):
    """Probability of actually cooperating when each move flips with `noise`."""
    return np.asarray(prob, dtype=float) * (1 - 2 * noise) + noise


def joint(a, b):
    """Distribution over CC, CD, DC, DD from independent cooperation probabilities."""
    a, b = np.broadcast_arrays(a, b)
    return np.stack([a * b, a * (1 - b), (1 - a) * b, (1 - a) * (1 - b)], axis=-1)


def chain(x, y, noise=0.0):
    """Initial distribution and transition matrix, from the point of view of x."""
    start = joint(noisy(x.first, noise), noisy(y.first, noise))
    matrix = joint(noisy(x.p, noise), noisy(np.array(y.p)[SWAP], noise))
    return start, matrix


def state_frequencies(x, y, rounds=200, noise=0.0):
    """Expected fraction of rounds spent in each state."""
    current, matrix = chain(x, y, noise)
    total = np.zeros(4)
    for _ in range(rounds):
        total += current
        current = current @ matrix
    return total / rounds


def expected_payoffs(x, y, rounds=200, noise=0.0, payoffs=AXELROD):
    """Expected average payoff per round for x and for y."""
    freq = state_frequencies(x, y, rounds, noise)
    return float(freq @ payoffs.vector), float(freq[SWAP] @ payoffs.vector)


def simulate(x, y, rounds, noise, rng, payoffs=AXELROD):
    """Play one match move by move. Used to cross-check the exact computation."""
    totals = np.zeros(2)
    state = None
    for _ in range(rounds):
        if state is None:
            px, py = x.first, y.first
        else:
            px, py = x.p[state], y.p[SWAP[state]]
        cx = rng.random() < noisy(px, noise)
        cy = rng.random() < noisy(py, noise)
        state = 2 * (not cx) + (not cy)
        totals += payoffs.vector[[state, SWAP[state]]]
    return tuple(totals / rounds)


def payoff_matrix(strategies, rounds=200, noise=0.0, payoffs=AXELROD):
    """A[i, j]: average payoff of strategy i against strategy j."""
    return np.array([
        [expected_payoffs(x, y, rounds, noise, payoffs)[0] for y in strategies]
        for x in strategies
    ])
