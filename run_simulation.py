"""
Run multiple evolutionary simulations with varying environmental conditions.

This script:
1. Initializes an evolution engine
2. Runs 50 simulations with random environments
3. Combines all data into a single dataset
4. Saves the dataset for analysis
"""

import pandas as pd
from simulation.environment import Environment
from simulation.evolution_engine import EvolutionEngine

print("=" * 60)
print("EVOLUTION SIMULATION")
print("=" * 60)

# Initialize evolution engine
engine = EvolutionEngine(pop_size=100, mutation_strength=0.05)

# Run multiple simulations
all_data = []
num_simulations = 50
generations_per_sim = 100

print(f"\nRunning {num_simulations} simulations...")
print(f"Population size: {engine.pop_size}")
print(f"Mutation strength: {engine.mutation_strength}")
print(f"Generations per simulation: {generations_per_sim}")
print("\nProgress:")

for sim_id in range(num_simulations):
    # Create random environment
    env = Environment.random()
    
    # Run simulation
    df = engine.run(env, generations=generations_per_sim, sim_id=sim_id)
    all_data.append(df)
    
    # Progress update
    print(f"  Sim {sim_id:02d} | "
          f"predator={env.predator_density:.2f} | "
          f"resource={env.resource_abundance:.2f}")

# Combine all data
print("\nCombining data...")
dataset = pd.concat(all_data, ignore_index=True)

# Save dataset
output_path = "data/evolution_dataset.csv"
dataset.to_csv(output_path, index=False)

print(f"\n{'=' * 60}")
print("SIMULATION COMPLETE")
print(f"{'=' * 60}")
print(f"Dataset saved: {output_path}")
print(f"Total rows: {dataset.shape[0]:,}")
print(f"Total columns: {dataset.shape[1]}")
print(f"\nColumns: {', '.join(dataset.columns.tolist())}")
print(f"\nSimulations: {dataset.sim_id.nunique()}")
print(f"Generations per sim: {dataset.generation.nunique()}")
print(f"\nSummary statistics:")
print(dataset[["size", "speed", "camouflage", "metabolism", "fitness"]].describe())
