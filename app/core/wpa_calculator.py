import math


class WPACalculator:
    """
    Win Probability Added (WPA) Calculator for KBO.
    Uses a simplified base matrix for win probabilities based on Inning, Score Diff, Outs, and Runners.
    """

    def __init__(self):
        # Simplified Win Expectancy Matrix (Inning -> Score Diff -> Win Prob)
        # This is a placeholder. In a real scenario, this should be loaded from historical KBO data.
        # Format: { Inning (1-9): { ScoreDiff (-5 to +5): WinProb } }
        # Score Diff = Home Score - Away Score (from Home team perspective)
        self.base_matrix = self._initialize_base_matrix()

    def _initialize_base_matrix(self):
        """
        Initializes a simplified win probability matrix.
        Assumption: 50% start. Score diff increases/decreases probability sigmoidally.
        Later innings amplify the impact of score difference.
        """
        matrix = {}
        for inning in range(1, 10):
            matrix[inning] = {}
            for diff in range(-10, 11):
                # Sigmoid function adjusted for inning pressure
                # As inning increases, small lead becomes more significant (slope increases)
                slope = 0.2 + (inning * 0.05)
                prob = 1 / (1 + math.exp(-slope * diff))
                matrix[inning][diff] = round(prob, 4)
        return matrix

    def calculate_win_probability(
        self, inning: int, is_top: bool, score_diff: int, outs: int, runners: tuple
    ):
        """
        Calculates win probability for the HOME team.

        Args:
            inning (int): Current inning (1-9+)
            is_top (bool): True if Top of inning, False if Bottom
            score_diff (int): Home Score - Away Score
            outs (int): 0, 1, 2
            runners (tuple): (runner_on_1st, runner_on_2nd, runner_on_3rd) - booleans

        Returns:
            float: Win Probability (0.0 to 1.0)
        """
        if inning > 9:
            inning = 9  # Cap at 9th inning logic for extras in this simplified model

        # Base probability from score difference
        base_prob = self.base_matrix.get(inning, {}).get(score_diff, 0.5)

        # Adjust for runners and outs (Simplified adjustments)
        # In a real model, this would be a lookup table (24 states per inning/score state)
        # Here we apply heuristics.

        adjustment = 0.0

        # Runners impact (Home team batting / Bottom of inning)
        runners_score = sum([1 for r in runners if r])

        # Leverage Index approximation
        if is_top:
            # Away team batting: Runners specific to Away team increase Away's win prob -> Decrease Home's
            # Outs increase Home's win prob
            if runners_score > 0:
                adjustment -= (
                    0.05 * runners_score * (1 + (inning / 10))
                )  # Runners hurt Home
            if outs > 0:
                adjustment += 0.02 * outs * (1 + (inning / 10))  # Outs help Home
        else:
            # Home team batting
            if runners_score > 0:
                adjustment += (
                    0.05 * runners_score * (1 + (inning / 10))
                )  # Runners help Home
            if outs > 0:
                adjustment -= 0.02 * outs * (1 + (inning / 10))  # Outs hurt Home

        # Cap adjustment to prevent overflow (simplistic)
        final_prob = base_prob + adjustment
        return max(0.0, min(1.0, final_prob))

    def calculate_wpa(self, before_state: dict, after_state: dict):
        """
        Calculates WPA for a specific play.
        """
        prob_before = self.calculate_win_probability(**before_state)
        prob_after = self.calculate_win_probability(**after_state)

        return prob_after - prob_before


# Example Usage
if __name__ == "__main__":
    calc = WPACalculator()

    # 9th Inning, Bottom, Tie Game (0 diff), 0 outs, No runners
    state_start = {
        "inning": 9,
        "is_top": False,
        "score_diff": 0,
        "outs": 0,
        "runners": (False, False, False),
    }

    # 9th Inning, Bottom, Tie Game, 0 outs, Runner on 1st (Hit)
    state_end = {
        "inning": 9,
        "is_top": False,
        "score_diff": 0,
        "outs": 0,
        "runners": (True, False, False),
    }

    wpa = calc.calculate_wpa(state_start, state_end)
    print(f"WPA for Single in 9th Bottom Tie Game: {wpa:.4f}")
