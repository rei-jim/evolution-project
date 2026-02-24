"""
Exploratory Data Analysis (EDA) of Evolution Simulation Data

This notebook analyzes:
1. How traits evolve over generations
2. How environment affects trait selection
3. Population diversity dynamics
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Load data
print("\nLoading data...")
df = pd.read_csv("data/evolution_dataset.csv")
print(f"Loaded {len(df):,} rows")
print(f"Columns: {', '.join(df.columns.tolist())}")

# ============================================================================
# 1. TRAIT EVOLUTION OVER GENERATIONS
# ============================================================================
print("\n" + "=" * 60)
print("1. TRAIT EVOLUTION OVER GENERATIONS")
print("=" * 60)

# Calculate mean traits per generation (across all simulations)
trait_means = df.groupby("generation")[["speed", "camouflage", "size", "metabolism", "fitness"]].mean()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Trait Evolution Over Generations (Average Across All Simulations)", fontsize=16)

for ax, trait in zip(axes.flat[:5], ["speed", "camouflage", "size", "metabolism", "fitness"]):
    ax.plot(trait_means[trait], linewidth=2)
    ax.set_title(f"Mean {trait.capitalize()}", fontsize=12)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean value")
    ax.grid(True, alpha=0.3)

# Remove empty subplot
axes.flat[5].remove()

plt.tight_layout()
plt.savefig("outputs/trait_evolution.png", dpi=150, bbox_inches='tight')
print("Saved: outputs/trait_evolution.png")

# Print trend summary
print("\nTrait trends (first generation → last generation):")
for trait in ["speed", "camouflage", "size", "metabolism", "fitness"]:
    first = trait_means[trait].iloc[0]
    last = trait_means[trait].iloc[-1]
    change = last - first
    pct_change = (change / first) * 100
    print(f"  {trait:12s}: {first:.4f} → {last:.4f} (change: {change:+.4f}, {pct_change:+.1f}%)")

# ============================================================================
# 2. EFFECT OF PREDATOR DENSITY ON TRAITS
# ============================================================================
print("\n" + "=" * 60)
print("2. EFFECT OF PREDATOR DENSITY ON TRAITS")
print("=" * 60)

# Create predator density groups
df["pred_group"] = pd.cut(df["predator_density"], bins=[0, 0.5, 1.0], labels=["low", "high"])

# Group by predator density and generation
pivot = df.groupby(["pred_group", "generation"])[["speed", "camouflage"]].mean().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Effect of Predator Density on Defensive Traits", fontsize=16)

for ax, trait in zip(axes, ["speed", "camouflage"]):
    for group, gdf in pivot.groupby("pred_group"):
        ax.plot(gdf["generation"], gdf[trait], label=f"{group} predators", linewidth=2)
    ax.set_title(f"{trait.capitalize()} Evolution", fontsize=12)
    ax.set_xlabel("Generation")
    ax.set_ylabel(f"Mean {trait}")
    ax.legend(title="Predator density")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/predator_effect.png", dpi=150, bbox_inches='tight')
print("Saved: outputs/predator_effect.png")

# Statistical summary
print("\nFinal generation trait values by predator density:")
final_gen = df[df["generation"] == df["generation"].max()]
summary = final_gen.groupby("pred_group")[["speed", "camouflage"]].mean()
print(summary)

# ============================================================================
# 3. TRAIT VARIANCE OVER TIME (DIVERSITY)
# ============================================================================
print("\n" + "=" * 60)
print("3. POPULATION DIVERSITY (TRAIT VARIANCE)")
print("=" * 60)

# Calculate variance per generation
trait_variance = df.groupby("generation")[["speed", "camouflage", "size", "metabolism"]].var()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Trait Variance Over Generations (Population Diversity)", fontsize=16)

for ax, trait in zip(axes.flat, ["speed", "camouflage", "size", "metabolism"]):
    ax.plot(trait_variance[trait], linewidth=2, color='darkred')
    ax.set_title(f"{trait.capitalize()} Variance", fontsize=12)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Variance")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/trait_variance.png", dpi=150, bbox_inches='tight')
print("Saved: outputs/trait_variance.png")

print("\nVariance trends (first → last generation):")
for trait in ["speed", "camouflage", "size", "metabolism"]:
    first = trait_variance[trait].iloc[0]
    last = trait_variance[trait].iloc[-1]
    change = last - first
    print(f"  {trait:12s}: {first:.6f} → {last:.6f} (change: {change:+.6f})")

# ============================================================================
# 4. FITNESS DISTRIBUTION
# ============================================================================
print("\n" + "=" * 60)
print("4. FITNESS DISTRIBUTION")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Fitness distribution at different generations
for gen in [0, 25, 50, 75, 99]:
    gen_data = df[df["generation"] == gen]["fitness"]
    axes[0].hist(gen_data, bins=30, alpha=0.5, label=f"Gen {gen}")

axes[0].set_title("Fitness Distribution Across Generations", fontsize=12)
axes[0].set_xlabel("Fitness")
axes[0].set_ylabel("Frequency")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Mean fitness over time
mean_fitness = df.groupby("generation")["fitness"].mean()
axes[1].plot(mean_fitness, linewidth=2, color='darkgreen')
axes[1].set_title("Mean Fitness Over Generations", fontsize=12)
axes[1].set_xlabel("Generation")
axes[1].set_ylabel("Mean Fitness")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/fitness_distribution.png", dpi=150, bbox_inches='tight')
print("Saved: outputs/fitness_distribution.png")

# ============================================================================
# 5. CORRELATION HEATMAP
# ============================================================================
print("\n" + "=" * 60)
print("5. TRAIT CORRELATIONS")
print("=" * 60)

# Calculate correlations for final generation
final_gen = df[df["generation"] == df["generation"].max()]
corr_cols = ["size", "speed", "camouflage", "metabolism", "fitness", 
             "predator_density", "resource_abundance"]
correlation = final_gen[corr_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title("Trait and Environment Correlations (Final Generation)", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=150, bbox_inches='tight')
print("Saved: outputs/correlation_heatmap.png")

print("\nStrongest positive correlations with fitness:")
fitness_corr = correlation["fitness"].sort_values(ascending=False)[1:4]
for trait, corr in fitness_corr.items():
    print(f"  {trait:20s}: {corr:+.3f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print("\nGenerated visualizations:")
print("  1. outputs/trait_evolution.png")
print("  2. outputs/predator_effect.png")
print("  3. outputs/trait_variance.png")
print("  4. outputs/fitness_distribution.png")
print("  5. outputs/correlation_heatmap.png")
