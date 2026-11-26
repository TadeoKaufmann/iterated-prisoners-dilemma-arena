"""Memory-one strategies for the iterated prisoner's dilemma.

A strategy is the probability of intending to cooperate on the first round, plus
that probability after each of the four possible previous rounds, seen from the
player's own side and ordered CC, CD, DC, DD (my move first, then theirs).
"""

from dataclasses import dataclass

STATES = ("CC", "CD", "DC", "DD")


@dataclass(frozen=True)
class Strategy:
    name: str
    first: float
    p: tuple

    def __post_init__(self):
        probabilities = (self.first, *self.p)
        if len(self.p) != 4 or not all(0.0 <= x <= 1.0 for x in probabilities):
            raise ValueError(f"{self.name}: need five probabilities in [0, 1]")


ALL_C = Strategy("AlwaysCooperate", 1, (1, 1, 1, 1))
ALL_D = Strategy("AlwaysDefect", 0, (0, 0, 0, 0))
TIT_FOR_TAT = Strategy("TitForTat", 1, (1, 0, 1, 0))
SUSPICIOUS_TFT = Strategy("SuspiciousTFT", 0, (1, 0, 1, 0))
GENEROUS_TFT = Strategy("GenerousTFT", 1, (1, 1 / 3, 1, 1 / 3))
ANTI_TFT = Strategy("AntiTitForTat", 1, (0, 1, 0, 1))
ALTERNATOR = Strategy("Alternator", 0, (0, 0, 1, 1))
GRIM = Strategy("GrimTrigger", 1, (1, 0, 0, 0))
PAVLOV = Strategy("Pavlov", 1, (1, 0, 0, 1))
JOSS = Strategy("Joss", 1, (0.9, 0, 0.9, 0))
# Press and Dyson's extortionate zero-determinant strategy with factor 3,
# defined for the payoffs T, R, P, S = 5, 3, 1, 0.
EXTORT_3 = Strategy("Extort3", 1, (11 / 13, 1 / 2, 7 / 26, 0))
RANDOM = Strategy("Random", 0.5, (0.5, 0.5, 0.5, 0.5))

FIELD = (
    ALL_C, ALL_D, TIT_FOR_TAT, SUSPICIOUS_TFT, GENEROUS_TFT, ANTI_TFT,
    ALTERNATOR, GRIM, PAVLOV, JOSS, EXTORT_3, RANDOM,
)
