import numpy as np
import time
import sys
sys.path.insert(0, ".")

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

from simulation.creature import Creature
from simulation.environment import Environment
from simulation.evolution_engine import EvolutionEngine

# --- Config ---
POP_SIZE     = 100
GENERATIONS  = 100
MUTATION_STR = 0.05
ENV          = Environment(predator_density=0.7, resource_abundance=0.5)
DELAY        = 0.1   # seconds between generations (set 0 for max speed)

console = Console()
engine  = EvolutionEngine(pop_size=POP_SIZE, mutation_strength=MUTATION_STR)

def make_bar(value: float, width: int = 20) -> str:
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)

def trait_color(value: float) -> str:
    if value > 0.7: return "bright_green"
    if value > 0.4: return "yellow"
    return "red"

def build_table(gen, history, population):
    means = {k: np.mean([getattr(c, k) for c in population])
             for k in ["speed","camouflage","size","metabolism","fitness"]}

    table = Table(title=f"Generation [bold cyan]{gen}[/bold cyan] / {GENERATIONS}",
                  show_header=True, header_style="bold magenta",
                  border_style="dim", min_width=60)

    table.add_column("Trait",    style="bold white",  width=14)
    table.add_column("Mean",     justify="right",     width=7)
    table.add_column("Bar",                           width=24)
    table.add_column("Trend",    justify="right",     width=10)

    traits = [
        ("speed",      "⚡"),
        ("camouflage", "🫥"),
        ("size",       "📏"),
        ("metabolism", "🔥"),
        ("fitness",    "💪"),
    ]

    for key, icon in traits:
        val = means[key]
        hist = history[key]
        trend = ""
        if len(hist) > 1:
            delta = val - hist[-2]
            trend = f"[green]+{delta:.3f}[/green]" if delta > 0 else f"[red]{delta:.3f}[/red]"

        bar_width = min(int(val * 20), 20)
        bar = f"[{trait_color(val)}]{make_bar(val)}[/{trait_color(val)}]"

        table.add_row(f"{icon} {key}", f"{val:.3f}", bar, trend)
        history[key].append(val)

    return table, means

def build_env_panel():
    return Panel(
        f"[yellow]Predator density:[/yellow]    {ENV.predator_density:.2f}  "
        f"{'█' * int(ENV.predator_density * 15)}\n"
        f"[green]Resource abundance:[/green]  {ENV.resource_abundance:.2f}  "
        f"{'█' * int(ENV.resource_abundance * 15)}",
        title="Environment", border_style="dim cyan"
    )

# --- Run ---
population = [Creature() for _ in range(POP_SIZE)]
history    = {"speed":[], "camouflage":[], "size":[], "metabolism":[], "fitness":[]}

with Live(console=console, refresh_per_second=20, screen=False) as live:
    for gen in range(GENERATIONS):
        for creature in population:
            creature.fitness = engine.compute_fitness(creature, ENV)

        table, means = build_table(gen, history, population)
        env_panel    = build_env_panel()

        survivors = sum(1 for c in population if c.fitness >= np.median(
                        [x.fitness for x in population]))

        stats = Panel(
            f"[cyan]Population:[/cyan] {POP_SIZE}    "
            f"[green]Survivors:[/green] {survivors}    "
            f"[magenta]Mutation:[/magenta] {MUTATION_STR}    "
            f"[yellow]Gen:[/yellow] {gen+1}/{GENERATIONS}",
            border_style="dim"
        )

        live.update(Columns([table, env_panel]))
        time.sleep(DELAY)

        parents    = engine.select_parents(population)
        population = engine.reproduce(parents)

console.print("\n[bold green]✓ Simulation complete![/bold green]")
console.print(f"Final mean fitness: [cyan]{np.mean([c.fitness for c in population]):.4f}[/cyan]")