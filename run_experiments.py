"""Regenerate every table and figure: python run_experiments.py"""

import csv
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image

from ipd.game import AXELROD, expected_payoffs, payoff_matrix
from ipd.learning import train
from ipd.population import basins, fixation_probability, replicator, spatial
from ipd.strategies import ALL_D, FIELD, TIT_FOR_TAT, Strategy

ROUNDS = 200
NOISE = 0.01
NAMES = [s.name for s in FIELD]
N = len(FIELD)
COLOURS = list(plt.get_cmap("tab20").colors[:N])
RESULTS, FIGURES = Path("results"), Path("figures")


def save_csv(name, header, rows):
    with open(RESULTS / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def tournament():
    A = payoff_matrix(FIELD, ROUNDS, NOISE)
    order = np.argsort(-A.mean(axis=1))
    save_csv("payoff_matrix.csv", ["strategy", *NAMES],
             [[NAMES[i], *np.round(A[i], 4)] for i in range(N)])

    fig, ax = plt.subplots(figsize=(9, 7.5))
    image = ax.imshow(A[np.ix_(order, order)], cmap="viridis", vmin=0, vmax=5)
    ax.set_xticks(range(N), [NAMES[i] for i in order], rotation=45, ha="right")
    ax.set_yticks(range(N), [f"{NAMES[i]}  ({A[i].mean():.2f})" for i in order])
    for r, c in itertools.product(range(N), range(N)):
        v = A[order[r], order[c]]
        ax.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=7,
                color="white" if v < 3 else "black")
    ax.set_title(f"Payoff of row against column ({ROUNDS} rounds, {NOISE:.0%} noise)\n"
                 "rows sorted by tournament average")
    fig.colorbar(image, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES / "payoff_matrix.png", dpi=140)
    plt.close(fig)
    return A


def noise_sweep():
    levels = [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    scores = np.array([payoff_matrix(FIELD, ROUNDS, e).mean(axis=1) for e in levels])
    ranks = (-scores).argsort(axis=1).argsort(axis=1) + 1
    save_csv("noise_sweep.csv", ["noise", *NAMES],
             [[e, *np.round(s, 4)] for e, s in zip(levels, scores)])
    save_csv("noise_ranks.csv", ["noise", *NAMES],
             [[e, *r] for e, r in zip(levels, ranks)])

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(NAMES):
        ax.plot(range(len(levels)), ranks[:, i], marker="o", color=COLOURS[i], lw=2)
        ax.text(len(levels) - 0.85, ranks[-1, i], name, va="center", fontsize=9)
    ax.set_xticks(range(len(levels)), [f"{e:.1%}" for e in levels])
    ax.set_yticks(range(1, N + 1))
    ax.invert_yaxis()
    ax.set_xlim(-0.3, len(levels) + 0.8)
    ax.set_xlabel("probability that a move is flipped")
    ax.set_ylabel("tournament rank")
    ax.set_title("How the ranking changes with noise")
    fig.tight_layout()
    fig.savefig(FIGURES / "noise_ranking.png", dpi=140)
    plt.close(fig)


def evolution(A):
    path = replicator(A, np.full(N, 1 / N), steps=6000, dt=0.05)
    won = basins(A)
    save_csv("evolution.csv", ["strategy", "final_share_uniform_start", "basin_share"],
             [[n, round(path[-1, i], 4), round(won[i], 4)] for i, n in enumerate(NAMES)])

    time = np.arange(len(path)) * 0.05
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.stackplot(time, path.T, colors=COLOURS, labels=NAMES)
    ax.set_xlim(0, time[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel("evolutionary time")
    ax.set_ylabel("population share")
    ax.set_title("Replicator dynamics from a uniform population")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "replicator.png", dpi=140)
    plt.close(fig)


def invasion(A):
    size = 50
    rho = np.array([[np.nan if i == j else fixation_probability(A, i, j, size) * size
                     for j in range(N)] for i in range(N)])
    save_csv("fixation_vs_neutral.csv", ["mutant\\resident", *NAMES],
             [[NAMES[i], *np.round(rho[i], 4)] for i in range(N)])

    fig, ax = plt.subplots(figsize=(9, 7.5))
    shown = np.log10(np.clip(rho, 1e-2, 1e1))
    image = ax.imshow(shown, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(N), NAMES, rotation=45, ha="right")
    ax.set_yticks(range(N), NAMES)
    ax.set_xlabel("resident population")
    ax.set_ylabel("single mutant")
    ax.set_title(f"Fixation probability relative to neutral drift (Moran, N = {size})\n"
                 "red: selection favours the invasion")
    bar = fig.colorbar(image, shrink=0.8, ticks=[-1, 0, 1])
    bar.ax.set_yticklabels(["0.1x or less", "1x", "10x or more"])
    fig.tight_layout()
    fig.savefig(FIGURES / "invasion.png", dpi=140)
    plt.close(fig)


def space(A):
    frames = spatial(A, size=80, generations=60, seed=3)
    shares = np.array([np.bincount(f.ravel(), minlength=N) / f.size for f in frames])
    save_csv("spatial_shares.csv", ["generation", *NAMES],
             [[g, *np.round(s, 4)] for g, s in enumerate(shares)])
    palette = (np.array(COLOURS) * 255).astype(np.uint8)
    images = [Image.fromarray(palette[f]).resize((400, 400), Image.NEAREST) for f in frames]
    images[0].save(FIGURES / "spatial.gif", save_all=True, append_images=images[1:],
                   duration=[800] + [150] * (len(images) - 2) + [2500], loop=0)

    picks = [0, 2, 5, 60]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2))
    for ax, g in zip(axes, picks):
        ax.imshow(frames[g], cmap=ListedColormap(COLOURS), vmin=-0.5, vmax=N - 0.5,
                  interpolation="nearest")
        ax.set_title(f"generation {g}")
        ax.axis("off")
    present = np.flatnonzero(shares[picks].max(axis=0) > 0.02)
    fig.legend(handles=[Patch(color=COLOURS[i], label=NAMES[i]) for i in present],
               loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(FIGURES / "spatial.png", dpi=140)
    plt.close(fig)


def best_response(grid_step=0.1):
    """Grid search for the memory-one strategy with the best average against the field."""
    values = np.round(np.arange(0, 1 + 1e-9, grid_step), 3)
    candidates = np.array(list(itertools.product(values, repeat=4)))
    results = []
    e = NOISE
    keep = lambda p: p * (1 - 2 * e) + e
    mine = keep(candidates)
    for first in (0.0, 1.0):
        total = np.zeros(len(candidates))
        for other in FIELD:
            theirs = keep(np.array(other.p)[[0, 2, 1, 3]])
            b0, a0 = keep(other.first), keep(first)
            current = np.tile([a0 * b0, a0 * (1 - b0), (1 - a0) * b0, (1 - a0) * (1 - b0)],
                              (len(candidates), 1))
            M = np.stack([mine * theirs, mine * (1 - theirs),
                          (1 - mine) * theirs, (1 - mine) * (1 - theirs)], axis=2)
            visits = np.zeros_like(current)
            for _ in range(ROUNDS):
                visits += current
                current = np.einsum("ns,nst->nt", current, M)
            total += visits @ AXELROD.vector / ROUNDS
        for p, score in zip(candidates, total / len(FIELD)):
            results.append((score, first, *p))
    results.sort(reverse=True)
    save_csv("best_memory_one.csv", ["score", "first", "p_CC", "p_CD", "p_DC", "p_DD"],
             [[round(r[0], 4), *r[1:]] for r in results[:20]])
    return results[0]


def learners(seeds=range(20)):
    rows = []
    for label, opponent in [("TitForTat", TIT_FOR_TAT), ("AlwaysDefect", ALL_D), ("itself", None)]:
        for seed in seeds:
            out = train(opponent=opponent, seed=seed)
            a, b = (out, opponent) if opponent is not None else out
            score_a, score_b = expected_payoffs(a, b, ROUNDS, 0.0)
            code = lambda s: "".join("C" if v else "D" for v in (s.first, *s.p))
            rows.append([label, seed, code(a), code(b) if opponent is None else "",
                         round(score_a, 4), round(score_b, 4)])
    save_csv("q_learning.csv",
             ["opponent", "seed", "policy_a", "policy_b", "score_a", "score_b"], rows)
    return rows


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    A = tournament()
    print("Tournament:")
    for i in np.argsort(-A.mean(axis=1)):
        print(f"  {NAMES[i]:16s} {A[i].mean():.3f}")
    noise_sweep()
    evolution(A)
    invasion(A)
    space(A)
    best = best_response()
    print(f"Best memory-one vs field: score {best[0]:.3f}, first {best[1]}, p {best[2:]}")
    rows = learners()
    for label in ("TitForTat", "AlwaysDefect", "itself"):
        scores = [r[4] for r in rows if r[0] == label]
        print(f"Q-learner vs {label:13s} mean score {np.mean(scores):.3f} "
              f"(min {min(scores):.2f}, max {max(scores):.2f})")
        if label == "itself":
            both = [(r[4] + r[5]) / 2 for r in rows if r[0] == label]
            print(f"  runs with mutual average above 2.5: {sum(v > 2.5 for v in both)}/{len(both)}")


if __name__ == "__main__":
    main()
