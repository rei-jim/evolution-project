# Evolution Project — Implementation Guide

## Project Overview

A stochastic evolutionary simulation where artificial organisms evolve under environmental pressures, with machine learning applied to analyze emergent adaptive dynamics.

**Stack:** Python · NumPy · Pandas · Matplotlib · Scikit-learn

---

## Folder Structure

```
evolution-project/
├── simulation/
│   ├── __init__.py
│   ├── creature.py
│   ├── environment.py
│   └── evolution_engine.py
├── data/
├── outputs/
├── notebooks/
│   ├── 01_eda.py
│   └── 02_ml_survival.py
└── run_simulation.py
```

---

## Setup

### 1. Create the project (Windows PowerShell)

```powershell
New-Item -ItemType Directory -Path "evolution-project/simulation"
New-Item -ItemType Directory -Path "evolution-project/data"
New-Item -ItemType Directory -Path "evolution-project/outputs"
New-Item -ItemType Directory -Path "evolution-project/notebooks"
cd evolution-project
New-Item simulation/__init__.py
New-Item simulation/creature.py
New-Item simulation/environment.py
New-Item simulation/evolution_engine.py
New-Item run_simulation.py
New-Item notebooks/01_eda.py
New-Item notebooks/02_ml_survival.py
```

### 2. Install dependencies

```powershell
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 3. Open in VS Code

```powershell
code .
```

---

## Week 1 — Build the Evolution Engine

### Day 1–2: `simulation/creature.py`

Each creature has 4 continuous traits in `[0, 1]`.

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class Creature:
    size: float = field(default_factory=lambda: np.random.uniform(0, 1))
    speed: float = field(default_factory=lambda: np.random.uniform(0, 1))
    camouflage: float = field(default_factory=lambda: np.random.uniform(0, 1))
    metabolism: float = field(default_factory=lambda: np.random.uniform(0, 1))
    fitness: float = 0.0

    def clip_traits(self):
        self.size = np.clip(self.size, 0, 1)
        self.speed = np.clip(self.speed, 0, 1)
        self.camouflage = np.clip(self.camouflage, 0, 1)
        self.metabolism = np.clip(self.metabolism, 0, 1)

    def mutate(self, mutation_strength: float = 0.05):
        self.size += np.random.normal(0, mutation_strength)
        self.speed += np.random.normal(0, mutation_strength)
        self.camouflage += np.random.normal(0, mutation_strength)
        self.metabolism += np.random.normal(0, mutation_strength)
        self.clip_traits()
```

**Verify:**
```powershell
python -c "from simulation.creature import Creature; c = Creature(); print(c)"
```

---

### Day 1–2: `simulation/environment.py`

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class Environment:
    predator_density: float = 0.7
    resource_abundance: float = 0.5

    @classmethod
    def random(cls):
        return cls(
            predator_density=np.random.uniform(0.1, 1.0),
            resource_abundance=np.random.uniform(0.1, 1.0),
        )
```

**Verify:**
```powershell
python -c "from simulation.environment import Environment; print(Environment())"
```

---

### Day 3–4: `simulation/evolution_engine.py` — Fitness Function

The fitness formula rewards speed and camouflage under high predator pressure, rewards metabolism when resources are abundant, and penalizes large size.

```
fitness = (speed × predator_density × 0.6)
        + (camouflage × predator_density × 0.9)
        + (metabolism × resource_abundance × 0.7)
        - (size × 0.4)
        + noise(0, 0.01)
```

```python
import numpy as np
import pandas as pd
from simulation.creature import Creature
from simulation.environment import Environment

class EvolutionEngine:
    def __init__(self, pop_size=100, mutation_strength=0.05):
        self.pop_size = pop_size
        self.mutation_strength = mutation_strength

    def compute_fitness(self, creature: Creature, env: Environment) -> float:
        f = (
            creature.speed * env.predator_density * 0.6
            + creature.camouflage * env.predator_density * 0.9
            + creature.metabolism * env.resource_abundance * 0.7
            - creature.size * 0.4
        )
        f += np.random.normal(0, 0.01)
        return max(f, 1e-6)
```

**Verify:**
```powershell
python -c "from simulation.evolution_engine import EvolutionEngine; from simulation.environment import Environment; from simulation.creature import Creature; e=EvolutionEngine(); print(e.compute_fitness(Creature(), Environment()))"
```

---

### Day 5–6: Selection + Reproduction

Add these methods to `EvolutionEngine`:

```python
    def select_parents(self, population):
        fitnesses = np.array([c.fitness for c in population])
        probs = fitnesses / fitnesses.sum()
        indices = np.random.choice(len(population), size=len(population), p=probs)
        return [population[i] for i in indices]

    def reproduce(self, parents):
        children = []
        for parent in parents:
            child = Creature(
                size=parent.size,
                speed=parent.speed,
                camouflage=parent.camouflage,
                metabolism=parent.metabolism
            )
            child.mutate(self.mutation_strength)
            children.append(child)
        return children
```

> No crossover yet. Clone + mutate only. This isolates selection behavior.

---

### Day 7: Full `run()` Method

Add to `EvolutionEngine`:

```python
    def run(self, env: Environment, generations=100, sim_id=0):
        population = [Creature() for _ in range(self.pop_size)]
        records = []

        for gen in range(generations):
            for creature in population:
                creature.fitness = self.compute_fitness(creature, env)

            fitnesses = [c.fitness for c in population]
            threshold = np.median(fitnesses)

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

            parents = self.select_parents(population)
            population = self.reproduce(parents)

        return pd.DataFrame(records)
```

**Verify:**
```powershell
python -c "from simulation.evolution_engine import EvolutionEngine; from simulation.environment import Environment; e=EvolutionEngine(); df=e.run(Environment(),100,0); print(df.groupby('generation')['fitness'].mean().tail())"
```

---

## Week 2 — Make It Data-Science Ready

### Day 1–4: `run_simulation.py`

Runs 50 simulations with randomized environments and saves a single combined CSV.

```python
import pandas as pd
from simulation.environment import Environment
from simulation.evolution_engine import EvolutionEngine

all_data = []
engine = EvolutionEngine(pop_size=100, mutation_strength=0.05)

for sim_id in range(50):
    env = Environment.random()
    df = engine.run(env, generations=100, sim_id=sim_id)
    all_data.append(df)
    print(f"Sim {sim_id:02d} | predator={env.predator_density:.2f} resource={env.resource_abundance:.2f}")

dataset = pd.concat(all_data, ignore_index=True)
dataset.to_csv("data/evolution_dataset.csv", index=False)
print(f"\nDataset saved: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
```

**Run it:**
```powershell
python run_simulation.py
```

**Verify:**
```powershell
python -c "import pandas as pd; df=pd.read_csv('data/evolution_dataset.csv'); print(df.sim_id.nunique(), 'simulations,', len(df), 'rows'); print(df.columns.tolist())"
```

Expected output: `50 simulations, 500000 rows`

---

### Day 5–6: `notebooks/01_eda.py`

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/evolution_dataset.csv")

# Trait means over generations
trait_means = df.groupby("generation")[["speed","camouflage","size","metabolism","fitness"]].mean()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Trait Evolution Over Generations", fontsize=14)
for ax, trait in zip(axes.flat, ["speed", "camouflage", "size", "metabolism"]):
    ax.plot(trait_means[trait])
    ax.set_title(f"Mean {trait}")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean value")
plt.tight_layout()
plt.savefig("outputs/trait_evolution.png")
print("Saved: outputs/trait_evolution.png")

# Predator density effect on speed and camouflage
df["pred_group"] = pd.cut(df["predator_density"], bins=[0, 0.5, 1.0], labels=["low","high"])
pivot = df.groupby(["pred_group","generation"])[["speed","camouflage"]].mean().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Effect of Predator Density on Traits", fontsize=14)
for ax, trait in zip(axes, ["speed", "camouflage"]):
    for group, gdf in pivot.groupby("pred_group"):
        ax.plot(gdf["generation"], gdf[trait], label=group)
    ax.set_title(trait)
    ax.set_xlabel("Generation")
    ax.legend(title="Predator density")
plt.tight_layout()
plt.savefig("outputs/predator_effect.png")
print("Saved: outputs/predator_effect.png")
```

**Run it:**
```powershell
python notebooks/01_eda.py
```

---

### Day 7: `notebooks/02_ml_survival.py`

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

df = pd.read_csv("data/evolution_dataset.csv")

features = ["size", "speed", "camouflage", "metabolism", "predator_density", "resource_abundance"]
target = "survived_next_gen"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]):.4f}")

# Feature importance plot
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
importances.plot(kind="bar", title="Feature Importances — Survival Prediction", figsize=(8,5))
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
print("Saved: outputs/feature_importance.png")
```

**Run it:**
```powershell
python notebooks/02_ml_survival.py
```

---

## Checklist

### Week 1
- [ ] `creature.py` — Creature dataclass with 4 traits + mutate()
- [ ] `environment.py` — Environment dataclass with random() factory
- [ ] `evolution_engine.py` — compute_fitness(), select_parents(), reproduce(), run()
- [ ] Single simulation runs 100 generations without errors
- [ ] Traits visibly shift toward high-fitness values over generations

### Week 2
- [ ] `run_simulation.py` — 50 simulations with randomized environments
- [ ] `data/evolution_dataset.csv` exists with 500,000 rows
- [ ] `outputs/trait_evolution.png` generated
- [ ] `outputs/predator_effect.png` generated
- [ ] `outputs/feature_importance.png` generated
- [ ] ROC-AUC above 0.75

---

## Key Concepts

| Term | What it means in this project |
|---|---|
| Fitness | A score computed from traits + environment. Higher = more likely to survive. |
| Selection | Picking parents with probability proportional to fitness. Good traits get passed on. |
| Mutation | Small random noise added to traits each generation. Prevents convergence. |
| Mutation-selection balance | Mutation adds variation; selection removes it. The tension between them drives evolution. |
| Survival label | Whether an individual's fitness was above the median — used as the ML target. |
| Feature importance | Which traits the Random Forest found most predictive of survival. Should match the fitness formula. |

---

## Common Errors

**`ModuleNotFoundError: No module named 'simulation'`**
Run scripts from the `evolution-project/` root folder, not from inside `simulation/`.

**`ZeroDivisionError` in select_parents**
All fitnesses are zero or negative. Check `compute_fitness` returns values > 0. The `max(f, 1e-6)` guard prevents this.

**Traits not shifting over generations**
Mutation strength may be too high (try `0.02`) or population size too small (try `200`).

**CSV is empty**
Make sure `run_simulation.py` is run from the project root and the `data/` folder exists.
