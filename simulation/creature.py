from dataclasses import dataclass, field
import numpy as np


@dataclass
class Creature:
    # --- Original traits ---
    size:          float = field(default_factory=lambda: np.random.uniform(0, 1))
    speed:         float = field(default_factory=lambda: np.random.uniform(0, 1))
    camouflage:    float = field(default_factory=lambda: np.random.uniform(0, 1))
    metabolism:    float = field(default_factory=lambda: np.random.uniform(0, 1))

    # --- New anti-predator traits ---
    vigilance:     float = field(default_factory=lambda: np.random.uniform(0, 1))
    armor:         float = field(default_factory=lambda: np.random.uniform(0, 1))
    maneuverability: float = field(default_factory=lambda: np.random.uniform(0, 1))

    # --- New life-history traits ---
    fat_reserves:  float = field(default_factory=lambda: np.random.uniform(0, 1))
    repro_invest:  float = field(default_factory=lambda: np.random.uniform(0, 1))
    risk_taking:   float = field(default_factory=lambda: np.random.uniform(0, 1))

    # --- Runtime state (not heritable) ---
    fitness:       float = 0.0
    alive:         bool  = True

    # Trait names used for mutation, logging, and ML
    HERITABLE_TRAITS = [
        "size", "speed", "camouflage", "metabolism",
        "vigilance", "armor", "maneuverability",
        "fat_reserves", "repro_invest", "risk_taking",
    ]

    def clip_traits(self):
        for t in self.HERITABLE_TRAITS:
            setattr(self, t, float(np.clip(getattr(self, t), 0.0, 1.0)))

    def mutate(self, mutation_strength: float = 0.02):
        """
        Gaussian mutation on all heritable traits.
        Default strength lowered to 0.02 per Lande & Arnold guidance
        (weak mutation, moderate selection regime).
        """
        for t in self.HERITABLE_TRAITS:
            val = getattr(self, t) + np.random.normal(0, mutation_strength)
            setattr(self, t, val)
        self.clip_traits()

    def standardize(self, trait: str) -> float:
        """
        Return z-score of a trait assuming population mean ≈ 0.5, std ≈ 0.25.
        Used to keep selection gradients on a biologically plausible scale.
        """
        return (getattr(self, trait) - 0.5) / 0.25