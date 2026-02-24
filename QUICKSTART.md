# 🚀 Quick Start Guide

## You're Ready to Go!

The Evolution Project is fully implemented with UV virtual environment. Here's how to use it:

## ⚡ Fastest Way (Run Everything)

```powershell
# Activate the virtual environment (if not already active)
.venv\Scripts\activate

# Run the complete pipeline (simulations + analysis + ML)
python run_all.py
```

This will take 5-10 minutes and generate all visualizations.

## 📝 Step-by-Step (Recommended for Learning)

### 1. Activate Virtual Environment
```powershell
.venv\Scripts\activate
```

### 2. Generate Evolution Data
```powershell
python run_simulation.py
```
⏱️ Takes 2-5 minutes  
📊 Creates `data/evolution_dataset.csv` (500,000 rows)

### 3. Analyze Evolution Patterns
```powershell
python notebooks/01_eda.py
```
⏱️ Takes 30-60 seconds  
📈 Creates 5 visualization files in `outputs/`

### 4. Train ML Model
```powershell
python notebooks/02_ml_survival.py
```
⏱️ Takes 30-60 seconds  
🤖 Creates 4 ML analysis visualizations in `outputs/`

## 📊 What You'll Get

### Data Files
- `data/evolution_dataset.csv` - Full simulation dataset

### Visualization Files (9 total)
1. `trait_evolution.png` - How traits change over time
2. `predator_effect.png` - Impact of predators on traits
3. `trait_variance.png` - Population diversity
4. `fitness_distribution.png` - Fitness patterns
5. `correlation_heatmap.png` - Trait relationships
6. `confusion_matrix.png` - ML model performance
7. `feature_importance.png` - Which traits matter most
8. `survival_by_trait.png` - Survival probability curves
9. `environment_effect.png` - Environmental impacts

## 🔧 Customize Your Simulation

### Change Parameters

Edit `run_simulation.py`:
```python
# Larger population
engine = EvolutionEngine(pop_size=200, mutation_strength=0.05)

# More simulations
num_simulations = 100

# Longer evolution
generations_per_sim = 200
```

### Modify Fitness Function

Edit `simulation/evolution_engine.py`:
```python
def compute_fitness(self, creature: Creature, env: Environment) -> float:
    f = (
        creature.speed * env.predator_density * 0.6  # Adjust weights here
        + creature.camouflage * env.predator_density * 0.9
        + creature.metabolism * env.resource_abundance * 0.7
        - creature.size * 0.4
    )
    return max(f, 1e-6)
```

## ✅ Verify Installation

```powershell
# Test all components
python -c "from simulation.creature import Creature; from simulation.environment import Environment; from simulation.evolution_engine import EvolutionEngine; print('✅ All modules working!')"
```

## 📖 Next Steps

1. **Week 1**: Run default simulations, explore visualizations
2. **Week 2**: Modify parameters, experiment with fitness function
3. **Week 3**: Add new features (crossover, new traits, dynamic environments)
4. **Week 4**: Advanced ML (neural networks, clustering, time series)

## 🆘 Help

See `README.md` for detailed documentation.

## 🎯 Expected Results

- **ML Accuracy**: 75-80%
- **ROC-AUC**: 0.75-0.85
- **Top Features**: camouflage, predator_density, speed
- **Evolution Pattern**: Speed & camouflage increase, size decreases

---

**Ready?** Run `python run_all.py` to see it all in action! 🎉
