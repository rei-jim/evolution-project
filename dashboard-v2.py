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
st.caption("Biologically grounded — Holling Type II/III predation · Standardized selection gradients · 10 heritable traits")

# --------------------------------------------------------------------------
# to run: .venv\Scripts\streamlit.exe run dashboard-v2.py
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.header("Population")
pop_size      = st.sidebar.slider("Population size",        50,  500, 100, step=50)
generations   = st.sidebar.slider("Generations",            10,  200, 100, step=10)
mutation_str  = st.sidebar.slider("Mutation strength",    0.005, 0.15, 0.02, step=0.005)
num_sims      = st.sidebar.slider("Number of simulations",   1,   20,   5)

st.sidebar.header("Environment")
predator      = st.sidebar.slider("Predator density",       0.0,  1.0, 0.7, step=0.05)
resource      = st.sidebar.slider("Resource abundance",     0.0,  1.0, 0.5, step=0.05)
attack_rate   = st.sidebar.slider("Attack rate (a)",        0.05, 1.0, 0.3, step=0.05)
handling_time = st.sidebar.slider("Handling time (h)",      0.01, 0.5, 0.1, step=0.01)
func_response = st.sidebar.radio("Holling curve type", ["II", "III"])
allow_crashes = st.sidebar.checkbox("Allow population crashes", value=True, 
                                     help="If enabled, populations can go extinct under harsh conditions. If disabled, population size is fixed (Wright-Fisher model).")

run_btn = st.sidebar.button("▶ Run Simulation", type="primary")

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
TRAIT_COLORS = {
    "speed":           "#00ff88",
    "camouflage":      "#00aaff",
    "size":            "#ff6644",
    "metabolism":      "#ffaa00",
    "vigilance":       "#cc88ff",
    "armor":           "#ff88cc",
    "maneuverability": "#44ffdd",
    "fat_reserves":    "#ffdd44",
    "repro_invest":    "#ff4488",
    "risk_taking":     "#88ff44",
}

TRAIT_GROUPS = {
    "Anti-Predator":  ["speed", "camouflage", "vigilance", "armor", "maneuverability"],
    "Life-History":   ["metabolism", "fat_reserves", "repro_invest", "risk_taking"],
    "Costly":         ["size"],
}

def caption(text: str):
    st.markdown(
        f"""<div style="background:#1a1a2e;border-left:3px solid #4a9eff;
        border-radius:4px;padding:10px 16px;margin-top:-8px;margin-bottom:24px;
        font-size:0.85rem;color:#a0aec0;line-height:1.6;">{text}</div>""",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
if run_btn:
    engine = EvolutionEngine(pop_size=pop_size, mutation_strength=mutation_str)
    env    = Environment(
        predator_density=predator,
        resource_abundance=resource,
        attack_rate=attack_rate,
        handling_time=handling_time,
        functional_response=func_response,
    )

    progress_bar = st.progress(0, text="Running simulations...")
    all_data = []
    extinctions = []

    for sim_id in range(num_sims):
        df = engine.run(env, generations=generations, sim_id=sim_id, allow_crashes=allow_crashes)
        all_data.append(df)
        
        # Check if this simulation went extinct
        if 'extinct' in df.columns and df['extinct'].any():
            extinct_gen = df[df['extinct'] == True]['generation'].min()
            extinctions.append((sim_id, extinct_gen))
        
        progress_bar.progress(
            (sim_id + 1) / num_sims,
            text=f"Simulation {sim_id+1}/{num_sims} complete"
        )

    progress_bar.empty()
    dataset   = pd.concat(all_data, ignore_index=True)
    all_traits = list(TRAIT_COLORS.keys())
    means     = dataset.groupby("generation")[all_traits + ["fitness", "population_size"]].mean().reset_index()
    variances = dataset.groupby("generation")[all_traits].var().reset_index()

    # Show extinction warnings
    if extinctions:
        st.warning(f"⚠️ {len(extinctions)}/{num_sims} simulations went extinct! " +
                   f"Extinction generations: {', '.join([f'Sim {s}: Gen {g}' for s, g in extinctions[:5]])}" +
                   (f" and {len(extinctions)-5} more..." if len(extinctions) > 5 else ""))
        st.info("💡 The population crashed due to high mortality. Try: lowering predator density, increasing resources, or disabling crashes to use a fixed-size model.")
    
    st.success(f"Done! {len(dataset):,} records · {num_sims} simulations · {generations} generations" +
               (f" · {len(extinctions)} extinct" if extinctions else ""))
    
    # Show key metrics
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Avg Population", f"{means['population_size'].mean():.1f}")
    with col_m2:
        st.metric("Final Fitness", f"{means['fitness'].iloc[-1]:.3f}")
    with col_m3:
        pop_stability = means['population_size'].std()
        st.metric("Pop Stability", f"{pop_stability:.1f}", delta="Lower = More Stable" if pop_stability > 5 else "Stable")
    with col_m4:
        fitness_gain = means['fitness'].iloc[-1] - means['fitness'].iloc[0]
        st.metric("Fitness Gain", f"{fitness_gain:+.3f}")

    # -----------------------------------------------------------------------
    # Section 1: Population dynamics
    # -----------------------------------------------------------------------
    st.subheader("Population Dynamics")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.line(means, x="generation", y="population_size",
                      title="Effective Population Size Over Time")
        fig.update_traces(line_color="#00ff88", line_width=3)
        fig.update_layout(
            template="plotly_dark",
            yaxis_title="Population Size",
            xaxis_title="Generation",
            yaxis=dict(range=[0, max(means["population_size"].max() * 1.1, pop_size * 1.1)])
        )
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.2)')
        st.plotly_chart(fig, use_container_width=True)

        avg_pop = means["population_size"].mean()
        K       = env.carrying_capacity
        min_pop = means["population_size"].min()
        caption(
            f"<b>What this shows:</b> How many individuals survive each generation after predation "
            f"and starvation/density-dependence mortality. "
            + (
                f"With <b>crashes enabled</b>, this is a true demographic population that can go extinct. "
                f"Each generation's size equals the number of survivors who reproduce (no refilling). "
                if allow_crashes else
                f"With <b>crashes disabled</b>, this shows the <i>effective breeding population</i> "
                f"but the next generation always starts with a fixed {pop_size} individuals (Wright-Fisher model). "
            ) +
            f"<br><br>"
            f"<b>Your result:</b> Average population = {avg_pop:.0f}, minimum = {min_pop:.0f} "
            f"(carrying capacity K = {K}). "
            + (
                f"Population dropped to near-extinction levels! High mortality is overwhelming reproduction."
                if min_pop < 10 and allow_crashes
                else
                f"Population is well below carrying capacity — predation is the dominant mortality source."
                if avg_pop < 0.7 * K
                else
                f"Population is near or above carrying capacity — density-dependent starvation is active."
            )
        )

    with col2:
        fig = px.line(means, x="generation", y="fitness",
                      title="Mean Reproductive Fitness Over Time")
        fig.update_traces(line_color="#ff44aa", line_width=3)
        fig.update_layout(
            template="plotly_dark",
            yaxis_title="Fitness",
            xaxis_title="Generation"
        )
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128,128,128,0.2)')
        st.plotly_chart(fig, use_container_width=True)

        start_fit = means["fitness"].iloc[0]
        end_fit   = means["fitness"].iloc[-1]
        plateau   = abs(means["fitness"].iloc[-1] - means["fitness"].iloc[-10]) < 0.02
        caption(
            f"<b>What this shows:</b> Average <i>reproductive fitness</i> — computed via "
            f"standardized selection gradients (Lande & Arnold 1983) on all 10 traits. "
            f"A rising curve = the population is adapting. A plateau = mutation–selection "
            f"balance has been reached (the biologically expected equilibrium). "
            f"<br><br>"
            f"<b>Your result:</b> Fitness {start_fit:.3f} → {end_fit:.3f}. "
            + (
                f"Plateau reached — population has found a stable adaptive strategy. "
                f"Try raising mutation strength to escape this local optimum."
                if plateau else
                f"Still climbing — run more generations to see the equilibrium."
            )
        )

    # -----------------------------------------------------------------------
    # Section 2: Anti-predator traits
    # -----------------------------------------------------------------------
    st.subheader("Anti-Predator Trait Evolution")
    col3, col4 = st.columns(2)

    anti_pred_traits = ["speed", "camouflage", "vigilance", "armor", "maneuverability"]
    ap_means = means[["generation"] + anti_pred_traits]

    with col3:
        fig = px.line(ap_means, x="generation",
                      y=["speed", "camouflage", "vigilance"],
                      title="Detection & Escape Traits",
                      color_discrete_map=TRAIT_COLORS)
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        dominant = max(["speed","camouflage","vigilance"],
                       key=lambda t: means[t].iloc[-1])
        caption(
            f"<b>What this shows:</b> Three traits that reduce the probability of being killed "
            f"by a predator. <i>Camouflage</i> lowers detection (reduces attack rate). "
            f"<i>Vigilance</i> also reduces effective attack rate by giving the prey early warning. "
            f"<i>Speed</i> reduces capture success once a predator has engaged. Under a "
            f"Holling Type II curve, all three act multiplicatively on the baseline kill rate λ."
            f"<br><br>"
            f"<b>Your result:</b> <i>{dominant}</i> is the dominant detection/escape trait at "
            f"generation {generations} under predator density {predator:.2f}. "
            + (
                f"High predator density is strongly selecting for anti-predator investment."
                if predator > 0.6 else
                f"Moderate predation pressure — traits are rising slowly."
            )
        )

    with col4:
        fig = px.line(ap_means, x="generation",
                      y=["armor", "maneuverability"],
                      title="Contact-Defense Traits",
                      color_discrete_map=TRAIT_COLORS)
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        caption(
            f"<b>What this shows:</b> <i>Armor</i> reduces kill probability <i>after</i> a "
            f"predator has attacked (the predator contacts but fails to kill). "
            f"<i>Maneuverability</i> reduces capture probability once being chased — "
            f"it operates after detection and contact, giving the prey a last-chance escape. "
            f"Together with speed and camouflage, these form a layered defense system."
            f"<br><br>"
            f"<b>Your result:</b> Armor ended at {means['armor'].iloc[-1]:.3f}, "
            f"maneuverability at {means['maneuverability'].iloc[-1]:.3f}. "
            + (
                f"Both are rising — multi-layer defense is being selected for."
                if means['armor'].iloc[-1] > 0.55 and means['maneuverability'].iloc[-1] > 0.55
                else
                f"These secondary defenses are developing slower — detection traits are "
                f"likely offering more fitness return per unit investment."
            )
        )

    # -----------------------------------------------------------------------
    # Section 3: Life-history traits
    # -----------------------------------------------------------------------
    st.subheader("Life-History Trait Evolution")
    col5, col6 = st.columns(2)

    with col5:
        fig = px.line(means, x="generation",
                      y=["metabolism", "fat_reserves", "risk_taking"],
                      title="Resource Acquisition Traits",
                      color_discrete_map=TRAIT_COLORS)
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        caption(
            f"<b>What this shows:</b> How the population manages energy. "
            f"<i>Metabolism</i> is rewarded when resources are abundant (current: {resource:.2f}) "
            f"but stabilizing selection prevents it from going to 1 (too costly). "
            f"<i>Fat reserves</i> buffer against starvation — especially important when the "
            f"population exceeds carrying capacity (K={env.carrying_capacity}). "
            f"<i>Risk-taking</i> boosts foraging when resources are scarce but increases "
            f"predation vulnerability — a classic predation–foraging trade-off."
            f"<br><br>"
            f"<b>Your result:</b> Risk-taking = {means['risk_taking'].iloc[-1]:.3f}. "
            + (
                f"High risk-taking is being selected — low resource abundance ({resource:.2f}) "
                f"is making bold foraging worth the danger."
                if means['risk_taking'].iloc[-1] > 0.55 and resource < 0.5 else
                f"Cautious foraging strategy — resources are sufficient that risk-taking "
                f"doesn't outweigh the predation cost."
            )
        )

    with col6:
        fig = px.line(means, x="generation",
                      y=["repro_invest", "size"],
                      title="Reproductive Investment & Body Size",
                      color_discrete_map=TRAIT_COLORS)
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        caption(
            f"<b>What this shows:</b> The classic life-history trade-off. "
            f"<i>Reproductive investment</i> increases offspring number but costs the adult "
            f"survival probability (an exponential mortality penalty). Stabilizing selection "
            f"(γ = −0.05) means intermediate investment is optimal. "
            f"<i>Size</i> has a directional penalty (β = −0.10) because larger individuals "
            f"are more conspicuous to predators, plus stabilizing selection around an "
            f"intermediate optimum."
            f"<br><br>"
            f"<b>Your result:</b> Size = {means['size'].iloc[-1]:.3f} "
            + (
                f"— decreased as expected under predation pressure."
                if means['size'].iloc[-1] < 0.5 else
                f"— still high; mutation may be counteracting selection on size."
            )
            + f" Repro investment = {means['repro_invest'].iloc[-1]:.3f}."
        )

    # -----------------------------------------------------------------------
    # Section 4: Population scatter (final gen)
    # -----------------------------------------------------------------------
    st.subheader("Final Generation — Population Cloud")
    last_gen = dataset[dataset["generation"] == dataset["generation"].max()]

    col7, col8 = st.columns(2)

    with col7:
        fig = px.scatter(
            last_gen, x="speed", y="camouflage",
            color="fitness", size="armor",
            title="Speed vs Camouflage (size = armor)",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        spread = last_gen["speed"].std() + last_gen["camouflage"].std()
        caption(
            f"<b>What this shows:</b> Each dot is one individual in the final generation. "
            f"Position = speed + camouflage values; color = fitness (yellow=high); "
            f"dot size = armor. A tight cluster = strong convergence on one strategy. "
            f"A spread cloud = diversity maintained by mutation or frequency-dependence."
            f"<br><br>"
            f"<b>Your result:</b> Spread = {spread:.3f}. "
            + (
                f"Tight cluster — the population has converged on a dominant anti-predator strategy."
                if spread < 0.5 else
                f"Diverse cloud — multiple anti-predator strategies are coexisting."
            )
        )

    with col8:
        fig = px.scatter(
            last_gen, x="metabolism", y="repro_invest",
            color="fitness", size="fat_reserves",
            title="Metabolism vs Repro Investment (size = fat reserves)",
            color_continuous_scale="Plasma",
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        caption(
            f"<b>What this shows:</b> The life-history trade-off space. "
            f"Individuals in the top-right invest heavily in both metabolism and reproduction "
            f"— high reward but high cost. The fitness color shows which combination actually "
            f"pays off under the current environment (resource abundance = {resource:.2f}). "
            f"Dot size = fat reserves — larger dots are better buffered against starvation."
            f"<br><br>"
            f"<b>Your result:</b> The high-fitness individuals (yellow) cluster around "
            f"metabolism ≈ {last_gen.nlargest(20,'fitness')['metabolism'].mean():.2f}, "
            f"repro_invest ≈ {last_gen.nlargest(20,'fitness')['repro_invest'].mean():.2f}."
        )

    # -----------------------------------------------------------------------
    # Section 5: Trait variance
    # -----------------------------------------------------------------------
    st.subheader("Trait Diversity (Variance)")
    fig = px.line(
        variances, x="generation", y=all_traits,
        title="Trait Variance Over Generations — All 10 Traits",
        color_discrete_map=TRAIT_COLORS,
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    avg_var       = variances[all_traits].iloc[-1].mean()
    most_diverse  = variances[all_traits].iloc[-1].idxmax()
    least_diverse = variances[all_traits].iloc[-1].idxmin()
    caption(
        f"<b>What this shows:</b> Genetic diversity for each trait over time. "
        f"High variance = many different strategies coexist (diverse population). "
        f"Low variance = convergence (everyone looks the same). Variance is set by the "
        f"balance between mutation (adds variation, strength = {mutation_str}) and "
        f"selection (removes it). Stabilizing selection on size, metabolism, and "
        f"repro_invest creates natural equilibrium points instead of driving traits to 0 or 1."
        f"<br><br>"
        f"<b>Your result:</b> Mean final variance = {avg_var:.4f}. "
        f"Most diverse trait: <i>{most_diverse}</i> — weakest selection or highest mutation replenishment. "
        f"Most converged trait: <i>{least_diverse}</i> — under strongest directional selection. "
        + (
            f"Healthy diversity maintained across all traits."
            if avg_var > 0.01 else
            f"Low diversity — increase mutation strength or add stabilizing selection."
        )
    )

    # -----------------------------------------------------------------------
    # Raw data
    # -----------------------------------------------------------------------
    with st.expander("View raw dataset"):
        st.dataframe(dataset.head(500))
        st.download_button(
            "Download full CSV", dataset.to_csv(index=False),
            "evolution_dataset.csv", "text/csv"
        )

else:
    st.info("👈 Configure parameters in the sidebar and click **Run Simulation** to start.")
    st.markdown("""
    **What's new in this version:**
    - 🦴 **10 heritable traits** — 5 anti-predator + 4 life-history + size
    - 🐺 **Holling Type II/III predation** — biologically realistic kill curves
    - 📊 **Standardized selection gradients** — based on Lande & Arnold (1983)
    - ⚖️ **Stabilizing selection** — prevents all traits converging to 1
    - 🍖 **Density-dependent starvation** — carrying capacity K scales with resources
    - 🎲 **Risk-taking trade-off** — bold foraging vs predation vulnerability
    - 💀 **Population crashes** — populations can go extinct under harsh conditions (toggle in sidebar)
    """)