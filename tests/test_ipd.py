import numpy as np
import pytest

from ipd.game import AXELROD, COURSE, Payoffs, expected_payoffs, payoff_matrix, simulate
from ipd.learning import train
from ipd.population import fixation_probability, replicator, spatial_step
from ipd.strategies import (ALL_C, ALL_D, ALTERNATOR, ANTI_TFT, EXTORT_3, FIELD,
                            GENEROUS_TFT, JOSS, PAVLOV, RANDOM, TIT_FOR_TAT)


def test_tft_against_always_defect():
    tft, alld = expected_payoffs(TIT_FOR_TAT, ALL_D, rounds=200)
    assert tft == pytest.approx(199 * 1 / 200)
    assert alld == pytest.approx((5 + 199 * 1) / 200)


def test_cooperators_cooperate():
    assert expected_payoffs(TIT_FOR_TAT, TIT_FOR_TAT) == pytest.approx((3, 3))


def test_noise_drags_tft_to_the_average_of_all_payoffs():
    x, _ = expected_payoffs(TIT_FOR_TAT, TIT_FOR_TAT, rounds=200_000, noise=0.01)
    assert x == pytest.approx((5 + 3 + 1 + 0) / 4, abs=0.01)


@pytest.mark.parametrize("x, y", [
    (TIT_FOR_TAT, ANTI_TFT), (TIT_FOR_TAT, ALTERNATOR), (ANTI_TFT, ALTERNATOR),
])
def test_course_exercise_pairs_average_one_half(x, y):
    a, b = expected_payoffs(x, y, rounds=4000, payoffs=COURSE)
    assert (a, b) == pytest.approx((0.5, 0.5), abs=1e-3)


@pytest.mark.parametrize("opponent", [ALL_C, RANDOM, PAVLOV, GENEROUS_TFT, JOSS])
def test_extortion_relation(opponent):
    """Press and Dyson: s_x - P = 3 (s_y - P) against any opponent."""
    sx, sy = expected_payoffs(EXTORT_3, opponent, rounds=200_000)
    assert sx - 1 == pytest.approx(3 * (sy - 1), abs=2e-3)


@pytest.mark.parametrize("noise", [0.0, 0.05])
def test_exact_matches_simulation(noise):
    rng = np.random.default_rng(1)
    for x, y in [(GENEROUS_TFT, JOSS), (PAVLOV, RANDOM), (ALTERNATOR, EXTORT_3)]:
        runs = np.array([simulate(x, y, 50, noise, rng) for _ in range(4000)])
        error = runs.std(axis=0) / np.sqrt(len(runs))
        exact = np.array(expected_payoffs(x, y, rounds=50, noise=noise))
        assert np.all(np.abs(runs.mean(axis=0) - exact) < 4 * error + 1e-9)


def test_payoffs_must_define_a_prisoners_dilemma():
    with pytest.raises(ValueError):
        Payoffs(T=3, R=5, P=1, S=0)


def test_neutral_fixation_is_one_over_n():
    A = payoff_matrix(FIELD)
    assert fixation_probability(A, 0, 1, size=40, selection=0) == pytest.approx(1 / 40)


def test_replicator_keeps_a_distribution_and_defection_beats_cooperation():
    A = payoff_matrix([ALL_C, ALL_D])
    path = replicator(A, [0.99, 0.01], steps=2000)
    assert np.allclose(path.sum(axis=1), 1)
    assert path[-1, 1] > 0.999


def test_uniform_grid_is_a_fixed_point():
    A = payoff_matrix([ALL_C, ALL_D])
    grid = np.zeros((10, 10), dtype=int)
    assert np.array_equal(spatial_step(A, grid), grid)


def test_learner_exploits_an_unconditional_cooperator():
    learned = train(opponent=ALL_C, episodes=30_000)
    assert expected_payoffs(learned, ALL_C)[0] == pytest.approx(AXELROD.T, abs=0.1)
