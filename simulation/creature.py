from dataclasses import dataclass, field
import numpy as np


@dataclass
class Creature:
    """A creature with 4 continuous traits in [0, 1]."""
    
    size: float = field(default_factory=lambda: np.random.uniform(0, 1))
    speed: float = field(default_factory=lambda: np.random.uniform(0, 1))
    camouflage: float = field(default_factory=lambda: np.random.uniform(0, 1))
    metabolism: float = field(default_factory=lambda: np.random.uniform(0, 1))
    fitness: float = 0.0

    def clip_traits(self):
        """Ensure all traits remain in [0, 1] range."""
        self.size = np.clip(self.size, 0, 1)
        self.speed = np.clip(self.speed, 0, 1)
        self.camouflage = np.clip(self.camouflage, 0, 1)
        self.metabolism = np.clip(self.metabolism, 0, 1)

    def mutate(self, mutation_strength: float = 0.05):
        """Apply Gaussian mutation to all traits."""
        self.size += np.random.normal(0, mutation_strength)
        self.speed += np.random.normal(0, mutation_strength)
        self.camouflage += np.random.normal(0, mutation_strength)
        self.metabolism += np.random.normal(0, mutation_strength)
        self.clip_traits()
