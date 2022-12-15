from numpy.random import exponential
from numpy import mean, percentile
import operator
from tsampling.root import RootThompsonSampling
from tsampling.priors import GammaPrior
from typing import List


class ExponentialExperiment(RootThompsonSampling):
    """ Class for Thompson Sampling with Exponential"""
    _default = {"shape": 0.001, "scale": 1000}
    _posterior = "gamma"

    def __init__(
        self, arms: int = None, priors: GammaPrior = None, labels: list = None
    ):
        super().__init__(arms, priors, labels)

    def choose_arm(self):
        """Choose which arm to pull
        
        Given the current posterior distributions this function will sample from
        the posterior and find the max theta of all the available options

        Parameters
        ----------

        Returns
        -------
        The max theta of all the available options
        """

        theta_est = {}
        for key, _ in self.posteriors.items():
            theta_est[key] = self._sample_posterior(1, key)
        return min(theta_est.items(), key=operator.itemgetter(1))[0]

    def add_rewards(self, results: List[dict]):
        """Takes in a list of dictionaries with the results and updates the Posterior
        distribution for the label.
        
        results = [{"label": "A", "reward": 1}, {"label":"B", "reward":0}]

        Parameters
        ----------
        results: List[dict] :
            

        """
        for result in results:
            self.posteriors[result["label"]]["shape"] += 1
            self.posteriors[result["label"]]["scale"] = round(
                1
                / ((1 / self.posteriors[result["label"]]["scale"]) + result["reward"]),
                8,
            )
        return self

    def get_distribution(self, size):
        """Simulates the posterior predictive distribution for a given
        label and returns the mean, and 95% credible interval.

        Parameters
        ----------
        size : int
            

        Returns
        -------
        Returns the mean, and 95% credible interval.
        """
        distribution_stats = []
        for k, _ in self.posteriors.items():
            pred_result = [
                int(
                    exponential(
                        scale=1
                        / (
                            self._avail_posteriors[self._posterior](
                                size=1, **self.posteriors[k]
                            )
                            + 1e-100
                        ),
                        size=1,
                    )
                )
                for _ in range(size)
            ]

            summary_stats = {
                "Label": k,
                "95% Credible Interval": (
                    round(percentile(pred_result, 2.5), 3),
                    round(percentile(pred_result, 97.5), 3),
                ),
                "mean": round(mean(pred_result), 3),
            }
            distribution_stats.append(summary_stats)
        return distribution_stats
