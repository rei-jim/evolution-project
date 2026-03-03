from dataclasses import dataclass
import numpy as np


@dataclass
class Environment:
    predator_density:   float = 0.7
    resource_abundance: float = 0.5

    # Holling Type II parameters
    attack_rate:    float = 0.3   # a  — baseline predator attack rate
    handling_time:  float = 0.1   # h  — time predator spends per prey

    # Carrying capacity (scales with resource abundance)
    base_carrying_capacity: int = 200

    # Holling curve type: "II" or "III"
    functional_response: str = "II"

    @property
    def carrying_capacity(self) -> int:
        return max(10, int(self.base_carrying_capacity * self.resource_abundance))

    def predation_lambda(self, population_size: int) -> float:
        """
        Compute per-prey baseline kill rate λ using the chosen
        Holling functional response.

        Returns λ_base, the expected number of attacks per individual
        from the entire predator population in one generation.
        """
        N = max(population_size, 1)
        P = self.predator_density
        a = self.attack_rate
        h = self.handling_time

        if self.functional_response == "III":
            # Sigmoidal — low predation at low prey density
            f = (a * N ** 2) / (1 + a * h * N ** 2)
        else:
            # Type II — saturating (default)
            f = (a * N) / (1 + a * h * N)

        total_kills = P * f
        return total_kills / N   # per-individual rate

    @classmethod
    def random(cls):
        return cls(
            predator_density=np.random.uniform(0.1, 1.0),
            resource_abundance=np.random.uniform(0.1, 1.0),
        )