# Evolution Project

A stochastic evolutionary simulation where artificial organisms evolve under environmental pressures, with machine learning applied to analyze emergent adaptive dynamics.

## 🎯 Project Overview

This project simulates the evolution of creatures with 4 continuous traits (size, speed, camouflage, metabolism) under varying environmental conditions (predator density, resource abundance). The simulation generates data that is then analyzed using machine learning to understand evolutionary dynamics.

## 🛠️ Setup (UV Virtual Environment)

The project is configured to use **UV** for fast dependency management.

### Prerequisites
- Python 3.9 or higher
- UV is already installed in this project

### Installation

The virtual environment and dependencies are already set up! To activate it:

```powershell
# Activate the virtual environment
.venv\Scripts\activate
```

To verify the installation:

```powershell
python -c "import numpy, pandas, matplotlib, sklearn; print('All dependencies installed!')"
```

## 📁 Project Structure

```
evolution-project/
├── simulation/           # Core simulation engine
│   ├── __init__.py
│   ├── creature.py      # Creature class with traits
│   ├── environment.py   # Environment conditions
│   └── evolution_engine.py  # Evolution logic
├── data/                # Generated datasets
├── outputs/             # Visualizations and results
├── notebooks/           # Analysis scripts
│   ├── 01_eda.py       # Exploratory Data Analysis
│   └── 02_ml_survival.py  # ML Survival Prediction
├── run_simulation.py    # Main simulation runner
├── pyproject.toml       # Project dependencies
└── README.md
```

## 🚀 Quick Start

### Step 1: Run Simulations

Generate the evolution dataset (50 simulations, 100 generations each):

```powershell
python run_simulation.py
```

This will:
- Run 50 simulations with randomized environments
- Generate 500,000 rows of data
- Save to `data/evolution_dataset.csv`
- Take approximately 2-5 minutes

### Step 2: Exploratory Data Analysis

Analyze how traits evolve over time:

```powershell
python notebooks/01_eda.py
```

This generates:
- `outputs/trait_evolution.png` - How traits change over generations
- `outputs/predator_effect.png` - Effect of predator density on traits
- `outputs/trait_variance.png` - Population diversity over time
- `outputs/fitness_distribution.png` - Fitness distributions
- `outputs/correlation_heatmap.png` - Trait correlations

### Step 3: Machine Learning Analysis

Train a Random Forest to predict survival:

```powershell
python notebooks/02_ml_survival.py
```

This generates:
- `outputs/confusion_matrix.png` - Model performance
- `outputs/feature_importance.png` - Which traits matter most
- `outputs/survival_by_trait.png` - Survival probability curves
- `outputs/environment_effect.png` - Environmental impact analysis

Expected ROC-AUC: **>0.75**

## 🧬 How It Works

### Fitness Function

Creatures are evaluated based on this fitness formula:

```python
fitness = (speed × predator_density × 0.6)
        + (camouflage × predator_density × 0.9)
        + (metabolism × resource_abundance × 0.7)
        - (size × 0.4)
        + noise(0, 0.01)
```

### Evolution Process

Each generation:
1. **Fitness Calculation**: Each creature gets a fitness score
2. **Selection**: Parents chosen proportional to fitness
3. **Reproduction**: Offspring created by cloning parents
4. **Mutation**: Small random changes (σ=0.05) applied to traits
5. **Logging**: All data recorded for analysis

### Key Parameters

- **Population size**: 100 creatures per generation
- **Generations**: 100 per simulation
- **Mutation strength**: 0.05 (standard deviation of Gaussian noise)
- **Trait range**: [0, 1] for all traits

## 📊 Research Questions

This project helps answer:

1. **Can ML recover the fitness structure?**  
   → Yes! Feature importance matches the fitness formula

2. **How does mutation strength affect stability?**  
   → Higher mutation maintains diversity but slows convergence

3. **Which traits dominate under different environments?**  
   → Speed and camouflage under high predators  
   → Metabolism under high resources

4. **Do ecosystems converge to similar strategies?**  
   → Similar environments lead to similar trait distributions

## 🔧 Customization

### Modify Simulation Parameters

Edit `run_simulation.py`:

```python
# Change population size
engine = EvolutionEngine(pop_size=200, mutation_strength=0.05)

# Run more simulations
num_simulations = 100

# More generations
generations_per_sim = 200
```

### Modify Fitness Function

Edit `simulation/evolution_engine.py` in the `compute_fitness` method:

```python
def compute_fitness(self, creature: Creature, env: Environment) -> float:
    # Customize the fitness formula here
    f = (
        creature.speed * env.predator_density * 0.8  # Increase speed importance
        + creature.camouflage * env.predator_density * 0.9
        # ... add your custom logic
    )
    return max(f, 1e-6)
```

## 📈 Expected Results

### Trait Evolution
- **Speed**: Increases under high predator pressure
- **Camouflage**: Increases under high predator pressure  
- **Size**: Decreases (due to penalty in fitness)
- **Metabolism**: Increases with resource abundance

### ML Performance
- **ROC-AUC**: 0.75 - 0.85
- **Top Features**: camouflage, predator_density, speed
- **Accuracy**: ~75-80%

## 🧪 Testing

Verify the installation works:

```powershell
# Test creature creation
python -c "from simulation.creature import Creature; c = Creature(); print(c)"

# Test environment
python -c "from simulation.environment import Environment; e = Environment.random(); print(e)"

# Test evolution engine
python -c "from simulation.evolution_engine import EvolutionEngine; from simulation.environment import Environment; e = EvolutionEngine(); df = e.run(Environment(), 10, 0); print(f'Generated {len(df)} rows')"
```

## 📚 Key Concepts

| Concept | Description |
|---------|-------------|
| **Fitness** | Score computed from traits + environment. Higher = more likely to survive |
| **Selection** | Parents chosen with probability proportional to fitness |
| **Mutation** | Random noise (σ=0.05) added to traits each generation |
| **Mutation-Selection Balance** | Mutation adds variation; selection removes it |
| **Survival Label** | Binary target: 1 if fitness ≥ median, 0 otherwise |
| **Feature Importance** | Which traits Random Forest found most predictive |

## 🐛 Troubleshooting

### Import Errors
**Problem**: `ModuleNotFoundError: No module named 'simulation'`  
**Solution**: Make sure you're running scripts from the project root directory

### UV Not Found
**Problem**: `uv: command not found`  
**Solution**: Use the full Python path or run:
```powershell
python -m uv venv
python -m uv pip install -e .
```

### Empty CSV
**Problem**: Dataset is empty or missing  
**Solution**: Ensure the `data/` folder exists and you're in the project root when running `run_simulation.py`

## 🎓 Learning Path

### Week 1: Understand the Simulation
1. Read the code in `simulation/` folder
2. Run `run_simulation.py` with 10 simulations
3. Inspect `data/evolution_dataset.csv`

### Week 2: Exploratory Analysis
1. Run `notebooks/01_eda.py`
2. Examine all generated plots
3. Form hypotheses about trait evolution

### Week 3: Machine Learning
1. Run `notebooks/02_ml_survival.py`
2. Analyze feature importance
3. Compare ML findings with fitness formula

### Week 4: Experimentation
1. Modify fitness function
2. Add new traits
3. Test different mutation rates
4. Implement crossover (currently only mutation)

## 🔬 Future Extensions

- [ ] Add sexual reproduction (crossover)
- [ ] Dynamic environments (change over time)
- [ ] Multiple species interaction
- [ ] Neural network fitness approximation
- [ ] Real-time visualization dashboard
- [ ] Genetic algorithm optimization
- [ ] Multi-objective optimization

## 📄 License

This is an educational project. Feel free to use and modify for learning purposes.

## 🤝 Contributing

This is a personal learning project. Suggestions and improvements welcome!

## 📖 References

- Evolutionary algorithms
- Genetic algorithms
- Stochastic simulation
- Trait-based ecology
- Feature importance analysis

---

**Author**: Evolution Project  
**Last Updated**: February 2026  
**Python**: 3.14+  
**Package Manager**: UV
