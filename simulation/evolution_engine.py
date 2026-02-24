import numpy as np
import pandas as pd
from simulation.creature import Creature
from simulation.environment import Environment


class EvolutionEngine:
    """Core engine for running evolutionary simulations."""
    
    def __init__(self, pop_size=100, mutation_strength=0.05):
        self.pop_size = pop_size
        self.mutation_strength = mutation_strength

    def compute_fitness(self, creature: Creature, env: Environment) -> float:
        """
        Compute fitness based on traits and environment.
        
        Formula rewards:
        - Speed and camouflage under high predator pressure
        - Metabolism when resources are abundant
        - Penalizes large size
        """
        f = (
            creature.speed * env.predator_density * 0.6
            + creature.camouflage * env.predator_density * 0.9
            + creature.metabolism * env.resource_abundance * 0.7
            - creature.size * 0.4
        )
        # Add small noise
        f += np.random.normal(0, 0.01)
        # Ensure fitness is positive
        return max(f, 1e-6)

    def select_parents(self, population):
        """Select parents proportional to fitness."""
        fitnesses = np.array([c.fitness for c in population])
        probs = fitnesses / fitnesses.sum()
        indices = np.random.choice(len(population), size=len(population), p=probs)
        return [population[i] for i in indices]

    def reproduce(self, parents):
        """Create offspring by cloning and mutating parents."""
        children = []
        for parent in parents:
            # Clone parent
            child = Creature(
                size=parent.size,
                speed=parent.speed,
                camouflage=parent.camouflage,
                metabolism=parent.metabolism
            )
            # Apply mutation
            child.mutate(self.mutation_strength)
            children.append(child)
        return children

    def run(self, env: Environment, generations=100, sim_id=0):
        """
        Run a complete simulation.
        
        Returns:
            DataFrame with individual-level data for each generation.
        """
        # Initialize population
        population = [Creature() for _ in range(self.pop_size)]
        records = []

        for gen in range(generations):
            # Compute fitness for all creatures
            for creature in population:
                creature.fitness = self.compute_fitness(creature, env)

            # Determine survival threshold (median fitness)
            fitnesses = [c.fitness for c in population]
            threshold = np.median(fitnesses)

            # Log data for this generation
            for creature in population:
                records.append({
                    "sim_id": sim_id,
                    "generation": gen,
                    "size": creature.size,
                    "speed": creature.speed,
                    "camouflage": creature.camouflage,
                    "metabolism": creature.metabolism,
                    "fitness": creature.fitness,
                    "survived_next_gen": int(creature.fitness >= threshold),
                    "predator_density": env.predator_density,
                    "resource_abundance": env.resource_abundance,
                })

            # Selection and reproduction
            parents = self.select_parents(population)
            population = self.reproduce(parents)

        return pd.DataFrame(records)
