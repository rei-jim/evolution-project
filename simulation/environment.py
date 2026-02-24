from dataclasses import dataclass
import numpy as np


@dataclass
class Environment:
    """Environmental conditions affecting creature fitness."""
    
    predator_density: float = 0.7
    resource_abundance: float = 0.5

    @classmethod
    def random(cls):
        """Create an environment with random conditions."""
        return cls(
            predator_density=np.random.uniform(0.1, 1.0),
            resource_abundance=np.random.uniform(0.1, 1.0),
        )
