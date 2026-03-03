import numpy as np
import pandas as pd
from math import exp

from simulation.creature import Creature
from simulation.environment import Environment


# ---------------------------------------------------------------------------
# Biologically grounded selection gradients (Lande & Arnold 1983).
# Values in [-0.3, 0.3] on standardized traits = moderate realistic selection.
# ---------------------------------------------------------------------------
BETA = {
    "speed":           0.15,   # faster → less likely to be caught
    "camouflage":      0.20,   # better camo → less detectable
    "metabolism":      0.10,   # higher metabolism → better resource use
    "size":           -0.10,   # larger → more conspicuous, costs energy
    "vigilance":       0.12,   # earlier predator detection
    "armor":           0.08,   # harder to kill once caught
    "maneuverability": 0.10,   # better escape once pursued
    "fat_reserves":    0.05,   # buffer against starvation
    "repro_invest":    0.08,   # more offspring (at a cost)
    "risk_taking":     0.06,   # foraging gain (environment-dependent)
}

# Quadratic (stabilizing) terms γ — small negative = stabilizing selection.
GAMMA = {
    "size":        -0.04,   # intermediate size is optimal
    "metabolism":  -0.03,   # intermediate metabolism is optimal
    "repro_invest":-0.05,   # too much investment is self-destructive
}

# Vulnerability modifiers (k values) for the Holling predation model.
K_SPEED          = 0.8    # speed reduces encounter success
K_CAMO           = 1.0    # camouflage reduces detection
K_SIZE           = 0.5    # large size increases conspicuousness
K_VIGILANCE      = 0.6    # vigilance reduces attack rate
K_ARMOR          = 0.7    # armor reduces kill-on-contact probability
K_MANEUVER       = 0.5    # maneuverability reduces capture after pursuit


class EvolutionEngine:

    def __init__(self, pop_size: int = 100, mutation_strength: float = 0.02):
        self.pop_size         = pop_size
        self.mutation_strength = mutation_strength

    # ------------------------------------------------------------------
    # 1. PREDATION STEP  (Holling functional response)
    # ------------------------------------------------------------------
    def apply_predation(self, population: list, env: Environment) -> list:
        """
        Kill individuals stochastically using a Holling functional response.
        Each creature's vulnerability is modulated by its anti-predator traits.
        Returns the list of survivors.
        """
        lambda_base = env.predation_lambda(len(population))

        survivors = []
        for c in population:
            # Vulnerability: exp(-protective_traits + costly_traits)
            v = exp(
                - K_SPEED    * c.speed
                - K_CAMO     * c.camouflage
                - K_VIGILANCE * c.vigilance
                + K_SIZE     * c.size          # large = more visible
            )
            # Kill probability: P(killed) = 1 - exp(-λ * v)
            p_kill = 1.0 - exp(-lambda_base * v)

            # Armor reduces kill probability once contacted
            p_kill *= exp(-K_ARMOR * c.armor)

            # Maneuverability further reduces capture (escape after pursuit)
            p_kill *= exp(-K_MANEUVER * c.maneuverability)

            if np.random.random() >= p_kill:
                survivors.append(c)

        return survivors

    # ------------------------------------------------------------------
    # 2. STARVATION STEP  (density-dependence + resource competition)
    # ------------------------------------------------------------------
    def apply_starvation(self, population: list, env: Environment) -> list:
        """
        Apply resource-competition mortality.
        If population exceeds carrying capacity, individuals with lower
        fat_reserves and higher risk_taking die preferentially.
        High risk_taking boosts survival when resources are scarce.
        """
        K = env.carrying_capacity
        N = len(population)
        if N <= K:
            return population

        # Each individual's starvation survival probability
        survival_probs = []
        for c in population:
            p = K / N
            # Fat reserves buffer against starvation
            p *= (1.0 + 0.3 * c.fat_reserves)
            # Risk-taking boosts foraging when resources are low
            if env.resource_abundance < 0.4:
                p *= (1.0 + 0.2 * c.risk_taking)
            survival_probs.append(min(p, 1.0))

        survivors = [
            c for c, p in zip(population, survival_probs)
            if np.random.random() < p
        ]
        return survivors

    # ------------------------------------------------------------------
    # 3. REPRODUCTIVE FITNESS  (standardized selection gradients)
    # ------------------------------------------------------------------
    def compute_reproductive_fitness(self, creature: Creature, env: Environment) -> float:
        """
        Reproductive fitness using standardized selection gradients
        (Lande & Arnold 1983). Returns a value > 0 used for parent
        selection weighted by reproductive output.

        Components:
          - Directional selection: β·z  for each trait
          - Stabilizing selection: γ·z² for selected traits
          - Resource modulation: metabolism and fat_reserves scaled by abundance
          - Trade-off penalty: repro_invest costs survival probability
        """
        fitness = 1.0  # base (relative fitness centered on 1)

        # Directional selection
        for trait, beta in BETA.items():
            z = creature.standardize(trait)
            # Modulate resource-dependent traits by environment
            if trait == "metabolism":
                beta = beta * env.resource_abundance
            elif trait == "risk_taking":
                # Risk-taking only pays off when resources are scarce
                beta = beta * (1.0 - env.resource_abundance)
            fitness += beta * z

        # Stabilizing / quadratic selection
        for trait, gamma in GAMMA.items():
            z = creature.standardize(trait)
            fitness += gamma * z ** 2

        # Reproductive investment multiplies offspring but costs the adult
        repro_bonus = 1.0 + 0.3 * creature.repro_invest
        repro_cost  = exp(-0.2 * creature.repro_invest)  # adult survival cost
        fitness     = fitness * repro_bonus * repro_cost

        # Small noise (environmental stochasticity)
        fitness += np.random.normal(0, 0.02)

        return max(fitness, 1e-6)

    # ------------------------------------------------------------------
    # 4. SELECTION + REPRODUCTION
    # ------------------------------------------------------------------
    def select_parents(self, population: list, target_size: int = None) -> list:
        """Select parents for next generation. If target_size is None, uses self.pop_size."""
        if target_size is None:
            target_size = self.pop_size
        if len(population) == 0:
            return []
        fitnesses = np.array([c.fitness for c in population])
        probs     = fitnesses / fitnesses.sum()
        indices   = np.random.choice(len(population), size=target_size, p=probs)
        return [population[i] for i in indices]

    def reproduce(self, parents: list) -> list:
        """
        Clone parent + mutate. Repro_invest scales expected offspring number
        (handled implicitly through fitness-proportional sampling).
        """
        children = []
        for parent in parents:
            child = Creature(
                **{t: getattr(parent, t) for t in Creature.HERITABLE_TRAITS}
            )
            child.mutate(self.mutation_strength)
            children.append(child)
        return children

    # ------------------------------------------------------------------
    # 5. FULL SIMULATION LOOP
    # ------------------------------------------------------------------
    def run(self, env: Environment, generations: int = 100, sim_id: int = 0, 
            allow_crashes: bool = True) -> pd.DataFrame:
        """
        Run evolution simulation.
        
        Args:
            env: Environment settings
            generations: Number of generations to simulate
            sim_id: Simulation identifier for tracking
            allow_crashes: If True, population can crash (demographic extinction).
                          If False, uses fixed pop_size (Wright-Fisher model).
        """
        population = [Creature() for _ in range(self.pop_size)]
        records    = []

        for gen in range(generations):
            # --- Mortality phase 1: predation ---
            survivors = self.apply_predation(population, env)

            # --- Mortality phase 2: starvation / density dependence ---
            survivors = self.apply_starvation(survivors, env)

            # --- Check for extinction ---
            if len(survivors) < 2 and allow_crashes:
                # Population has crashed - record final state and stop
                if len(survivors) > 0:
                    for c in survivors:
                        c.fitness = self.compute_reproductive_fitness(c, env)
                        row = {t: getattr(c, t) for t in Creature.HERITABLE_TRAITS}
                        row.update({
                            "sim_id":             sim_id,
                            "generation":         gen,
                            "fitness":            c.fitness,
                            "survived_next_gen":  0,  # extinction
                            "predator_density":   env.predator_density,
                            "resource_abundance": env.resource_abundance,
                            "population_size":    len(survivors),
                            "n_died_predation":   len(population) - len(survivors),
                            "extinct":            True,
                        })
                        records.append(row)
                break  # Stop simulation - population extinct

            # --- Compute reproductive fitness for survivors ---
            for c in survivors:
                c.fitness = self.compute_reproductive_fitness(c, env)

            # --- Record this generation ---
            all_fitnesses   = [c.fitness for c in survivors]
            fit_median      = np.median(all_fitnesses) if len(all_fitnesses) > 0 else 0
            n_died          = len(population) - len(survivors)

            for c in survivors:
                row = {t: getattr(c, t) for t in Creature.HERITABLE_TRAITS}
                row.update({
                    "sim_id":             sim_id,
                    "generation":         gen,
                    "fitness":            c.fitness,
                    "survived_next_gen":  int(c.fitness >= fit_median),
                    "predator_density":   env.predator_density,
                    "resource_abundance": env.resource_abundance,
                    "population_size":    len(survivors),
                    "n_died_predation":   n_died,
                    "extinct":            False,
                })
                records.append(row)

            # --- Reproduce for next generation ---
            if allow_crashes:
                # Dynamic population: survivors produce offspring (can crash)
                target_size = len(survivors)
            else:
                # Fixed population: always refill to pop_size (Wright-Fisher)
                target_size = self.pop_size
            
            parents    = self.select_parents(survivors, target_size=target_size)
            population = self.reproduce(parents)

        return pd.DataFrame(records)