import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
sys.path.insert(0, ".")

from simulation.creature import Creature
from simulation.environment import Environment
from simulation.evolution_engine import EvolutionEngine

st.set_page_config(page_title="Evolution Simulator", layout="wide")
st.title("🧬 Generative Evolution Simulator")

# --- Sidebar controls ---
st.sidebar.header("Parameters")
pop_size       = st.sidebar.slider("Population size",      50,  500, 100, step=50)
generations    = st.sidebar.slider("Generations",          10,  200, 100, step=10)
mutation_str   = st.sidebar.slider("Mutation strength",    0.01, 0.3, 0.05, step=0.01)
predator       = st.sidebar.slider("Predator density",     0.0,  1.0, 0.7,  step=0.05)
resource       = st.sidebar.slider("Resource abundance",   0.0,  1.0, 0.5,  step=0.05)
num_sims       = st.sidebar.slider("Number of simulations", 1,   20,  5)
run_btn        = st.sidebar.button("▶ Run Simulation", type="primary")

def caption(text: str):
    """Render a styled explanation block under a chart."""
    st.markdown(
        f"""
        <div style="
            background: #1a1a2e;
            border-left: 3px solid #4a9eff;
            border-radius: 4px;
            padding: 10px 16px;
            margin-top: -8px;
            margin-bottom: 24px;
            font-size: 0.85rem;
            color: #a0aec0;
            line-height: 1.6;
        ">{text}</div>
        """,
        unsafe_allow_html=True,
    )

if run_btn:
    engine = EvolutionEngine(pop_size=pop_size, mutation_strength=mutation_str)
    env    = Environment(predator_density=predator, resource_abundance=resource)

    progress_bar = st.progress(0, text="Running simulations...")
    all_data = []

    for sim_id in range(num_sims):
        df = engine.run(env, generations=generations, sim_id=sim_id)
        all_data.append(df)
        progress_bar.progress(
            (sim_id + 1) / num_sims,
            text=f"Simulation {sim_id+1}/{num_sims} done"
        )

    progress_bar.empty()
    dataset = pd.concat(all_data, ignore_index=True)
    means   = dataset.groupby("generation")[
                  ["speed", "camouflage", "size", "metabolism", "fitness"]
              ].mean().reset_index()
    variances = dataset.groupby("generation")[
                    ["speed", "camouflage", "size", "metabolism"]
                ].var().reset_index()

    st.success(f"Done! {len(dataset):,} individuals across {num_sims} simulations.")

    # ------------------------------------------------------------------ #
    # Row 1: Predator-response traits + Resource traits
    # ------------------------------------------------------------------ #
    st.subheader("Trait Evolution Over Generations")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(
            means, x="generation", y=["speed", "camouflage"],
            title="Predator-Response Traits",
            labels={"value": "Mean value", "variable": "Trait"},
            color_discrete_map={"speed": "#00ff88", "camouflage": "#00aaff"},
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # Dynamic interpretation
        final_speed = means["speed"].iloc[-1]
        final_camo  = means["camouflage"].iloc[-1]
        dominant    = "camouflage" if final_camo > final_speed else "speed"
        caption(
            f"<b>What this shows:</b> How <i>speed</i> and <i>camouflage</i> evolve under predator "
            f"pressure (currently set to <b>{predator:.2f}</b>). Both traits are directly rewarded "
            f"by the fitness function when predators are present, so you expect them to rise over "
            f"time. A rising curve means selection is actively pushing the population toward better "
            f"predator avoidance. "
            f"<br><br>"
            f"<b>Your result:</b> At generation {generations}, <i>{dominant}</i> is the dominant "
            f"predator-response trait "
            f"(speed={final_speed:.2f}, camouflage={final_camo:.2f}). "
            + (
                f"Camouflage dominates because its fitness weight (0.9) is higher than speed's (0.6) — "
                f"the simulation is correctly reflecting that hiding is more effective than running."
                if dominant == "camouflage"
                else
                f"Speed is unusually dominant here — check whether camouflage is being suppressed "
                f"by a trade-off with another trait."
            )
        )

    with col2:
        fig = px.line(
            means, x="generation", y=["size", "metabolism"],
            title="Resource & Cost Traits",
            labels={"value": "Mean value", "variable": "Trait"},
            color_discrete_map={"size": "#ff6644", "metabolism": "#ffaa00"},
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        final_size = means["size"].iloc[-1]
        final_meta = means["metabolism"].iloc[-1]
        caption(
            f"<b>What this shows:</b> <i>Size</i> is penalized by the fitness function (−0.4 "
            f"penalty), so you expect it to decrease over generations — smaller creatures survive "
            f"better. <i>Metabolism</i> is rewarded when resources are abundant "
            f"(current resource abundance: <b>{resource:.2f}</b>), so it should rise when food "
            f"is plentiful and stagnate when it's scarce. "
            f"<br><br>"
            f"<b>Your result:</b> Size ended at {final_size:.2f} "
            + (
                f"— it decreased as expected, confirming selection pressure against large body size."
                if final_size < 0.5
                else
                f"— it's still relatively high, which may indicate mutation is counteracting "
                f"selection, or the penalty isn't strong enough yet."
            )
            + f" Metabolism ended at {final_meta:.2f} "
            + (
                f"— healthy given resource abundance of {resource:.2f}."
                if final_meta > resource - 0.1
                else
                f"— lower than expected given resource abundance; other selection pressures "
                f"may be limiting it."
            )
        )

    # ------------------------------------------------------------------ #
    # Row 2: Fitness over time + Population scatter
    # ------------------------------------------------------------------ #
    st.subheader("Population Analysis")
    col3, col4 = st.columns(2)

    with col3:
        fig = px.line(
            means, x="generation", y="fitness",
            title="Mean Fitness Over Time",
        )
        fig.update_traces(line_color="#ff44aa")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        start_fit = means["fitness"].iloc[0]
        end_fit   = means["fitness"].iloc[-1]
        gain      = end_fit - start_fit
        plateau   = abs(means["fitness"].iloc[-1] - means["fitness"].iloc[-10]) < 0.01

        caption(
            f"<b>What this shows:</b> The overall health of the population over time. A rising "
            f"curve means the population is adapting — individuals are becoming better suited to "
            f"the environment. A plateau means the population has reached a local fitness peak "
            f"and mutation is balanced by selection (mutation–selection balance). "
            f"A declining curve would indicate the environment is too harsh or mutation is "
            f"too destructive. "
            f"<br><br>"
            f"<b>Your result:</b> Fitness went from {start_fit:.3f} → {end_fit:.3f} "
            f"(+{gain:.3f}). "
            + (
                f"The curve has plateaued — the population has converged to a stable adaptive "
                f"strategy for this environment. Try increasing mutation strength to escape "
                f"the local optimum."
                if plateau
                else
                f"Fitness is still climbing — the population hasn't fully adapted yet. "
                f"Try running more generations to see where it stabilizes."
            )
        )

    with col4:
        last_gen = dataset[dataset["generation"] == dataset["generation"].max()]
        fig = px.scatter(
            last_gen, x="speed", y="camouflage",
            color="fitness", size="metabolism",
            title="Final Generation — Speed vs Camouflage",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        cluster_x = last_gen["speed"].mean()
        cluster_y = last_gen["camouflage"].mean()
        spread    = last_gen["speed"].std() + last_gen["camouflage"].std()
        caption(
            f"<b>What this shows:</b> A snapshot of every individual in the final generation. "
            f"Each dot is one creature — its position shows its speed and camouflage values, "
            f"its color shows fitness (yellow = high, purple = low), and its size reflects "
            f"metabolism. A tight cluster means the population has converged on a single "
            f"strategy. A spread-out cloud means diversity is being maintained by mutation. "
            f"<br><br>"
            f"<b>Your result:</b> The population is centered around speed={cluster_x:.2f}, "
            f"camouflage={cluster_y:.2f}. "
            + (
                f"The cluster is tight (spread={spread:.2f}) — strong convergence, low diversity. "
                f"The population has locked onto one strategy."
                if spread < 0.4
                else
                f"The cloud is spread out (spread={spread:.2f}) — good diversity is being "
                f"maintained. Mutation is preventing the population from over-specializing."
            )
        )

    # ------------------------------------------------------------------ #
    # Row 3: Trait variance
    # ------------------------------------------------------------------ #
    st.subheader("Trait Diversity (Variance)")
    fig = px.line(
        variances, x="generation",
        y=["speed", "camouflage", "size", "metabolism"],
        title="Trait Variance Over Generations",
        labels={"value": "Variance", "variable": "Trait"},
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    avg_var      = variances[["speed","camouflage","size","metabolism"]].iloc[-1].mean()
    most_diverse = variances[["speed","camouflage","size","metabolism"]].iloc[-1].idxmax()
    least_diverse= variances[["speed","camouflage","size","metabolism"]].iloc[-1].idxmin()
    caption(
        f"<b>What this shows:</b> How much variation exists in each trait across the population "
        f"over time. High variance means the population is diverse — many different strategies "
        f"are present. Low variance means everyone looks similar — the population has converged. "
        f"Variance typically drops early as selection eliminates bad traits, then stabilizes "
        f"at a level set by mutation strength (currently <b>{mutation_str}</b>). "
        f"If variance hits zero, the population is genetically uniform — a warning sign that "
        f"it may be fragile to environmental change. "
        f"<br><br>"
        f"<b>Your result:</b> Average final variance is {avg_var:.4f}. "
        f"<i>{most_diverse}</i> is the most diverse trait — selection pressure on it is "
        f"relatively weak or mutation keeps replenishing it. "
        f"<i>{least_diverse}</i> has converged the most — it's under the strongest directional "
        f"selection. "
        + (
            f"Overall diversity is healthy — the population is not at risk of genetic uniformity."
            if avg_var > 0.01
            else
            f"Variance is very low across all traits — consider increasing mutation strength "
            f"to maintain diversity."
        )
    )

    # ------------------------------------------------------------------ #
    # Raw data
    # ------------------------------------------------------------------ #
    with st.expander("View raw dataset"):
        st.dataframe(dataset.head(500))
        st.download_button(
            "Download CSV", dataset.to_csv(index=False),
            "evolution_dataset.csv", "text/csv"
        )

else:
    st.info("👈 Set parameters in the sidebar and click **Run Simulation** to start.")