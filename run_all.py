"""
Quick Start Script - Run Complete Evolution Project Pipeline

This script runs the entire analysis pipeline:
1. Generate evolution dataset
2. Perform exploratory data analysis
3. Train and evaluate ML model

Usage:
    python run_all.py
"""

import subprocess
import sys
import os
from pathlib import Path

# Ensure we're in the project root
os.chdir(Path(__file__).parent)

print("=" * 70)
print("EVOLUTION PROJECT - COMPLETE PIPELINE")
print("=" * 70)

# Step 1: Run simulations
print("\n" + "=" * 70)
print("STEP 1: RUNNING SIMULATIONS")
print("=" * 70)
print("This will take 2-5 minutes...\n")

result = subprocess.run([sys.executable, "run_simulation.py"])
if result.returncode != 0:
    print("\n❌ Simulation failed!")
    sys.exit(1)

# Step 2: EDA
print("\n" + "=" * 70)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

result = subprocess.run([sys.executable, "notebooks/01_eda.py"])
if result.returncode != 0:
    print("\n❌ EDA failed!")
    sys.exit(1)

# Step 3: ML Analysis
print("\n" + "=" * 70)
print("STEP 3: MACHINE LEARNING ANALYSIS")
print("=" * 70)

result = subprocess.run([sys.executable, "notebooks/02_ml_survival.py"])
if result.returncode != 0:
    print("\n❌ ML analysis failed!")
    sys.exit(1)

# Success!
print("\n" + "=" * 70)
print("✅ PIPELINE COMPLETE!")
print("=" * 70)
print("\nGenerated files:")
print("  📊 Data:")
print("     - data/evolution_dataset.csv")
print("\n  📈 Visualizations:")
print("     - outputs/trait_evolution.png")
print("     - outputs/predator_effect.png")
print("     - outputs/trait_variance.png")
print("     - outputs/fitness_distribution.png")
print("     - outputs/correlation_heatmap.png")
print("     - outputs/confusion_matrix.png")
print("     - outputs/feature_importance.png")
print("     - outputs/survival_by_trait.png")
print("     - outputs/environment_effect.png")
print("\n" + "=" * 70)
print("Open the outputs folder to view all visualizations!")
print("=" * 70)
