"""Population dynamics built on a payoff matrix A[i, j]."""

import numpy as np


def replicator(A, start, steps=20000, dt=0.05):
    """Replicator dynamics x_i' = x_i (f_i - mean f), integrated multiplicatively
    so that shares stay positive and add up to one. Returns the whole trajectory."""
    x = np.asarray(start, dtype=float)
    path = np.empty((steps + 1, x.size))
    path[0] = x
    for t in range(1, steps + 1):
        fitness = A @ x
        x = x * np.exp(dt * (fitness - x @ fitness))
        x /= x.sum()
        path[t] = x
    return path


def basins(A, samples=2000, steps=4000, dt=0.1, seed=0):
    """From uniformly random initial populations, which strategy ends up with the
    largest share? Returns the fraction of starts won by each strategy."""
    rng = np.random.default_rng(seed)
    x = rng.dirichlet(np.ones(len(A)), size=samples)
    for _ in range(steps):
        fitness = x @ A.T
        mean = np.sum(x * fitness, axis=1, keepdims=True)
        x = x * np.exp(dt * (fitness - mean))
        x /= x.sum(axis=1, keepdims=True)
    return np.bincount(x.argmax(axis=1), minlength=len(A)) / samples


def fixation_probability(A, mutant, resident, size=50, selection=0.1):
    """Moran process: chance that a single mutant takes over a resident
    population. Fitness is 1 - selection + selection * average payoff, with no
    self-play. The neutral benchmark is 1 / size."""
    i = np.arange(1, size)  # current number of mutants
    f = (A[mutant, mutant] * (i - 1) + A[mutant, resident] * (size - i)) / (size - 1)
    g = (A[resident, mutant] * i + A[resident, resident] * (size - i - 1)) / (size - 1)
    f = 1 - selection + selection * f
    g = 1 - selection + selection * g
    return float(1 / (1 + np.sum(np.cumprod(g / f))))


NEIGHBOURS = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]


def spatial_step(A, grid):
    """Every cell plays its eight neighbours (wrapping around the edges), then
    copies the highest scorer in its neighbourhood, itself included."""
    score = sum(A[grid, np.roll(grid, shift, axis=(0, 1))] for shift in NEIGHBOURS)
    best_score, best = score.copy(), grid.copy()
    for shift in NEIGHBOURS:
        other_score = np.roll(score, shift, axis=(0, 1))
        better = other_score > best_score + 1e-12
        best_score = np.where(better, other_score, best_score)
        best = np.where(better, np.roll(grid, shift, axis=(0, 1)), best)
    return best


def spatial(A, size=80, generations=60, seed=0):
    rng = np.random.default_rng(seed)
    frames = [rng.integers(len(A), size=(size, size))]
    for _ in range(generations):
        frames.append(spatial_step(A, frames[-1]))
    return np.array(frames)
