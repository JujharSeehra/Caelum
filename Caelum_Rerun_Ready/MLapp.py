from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

st.set_page_config(
    page_title="Caelum Carbon Capture Optimizer",
    page_icon="🌍",
    layout="wide",
)

st.title("Caelum Carbon Capture Optimizer")
st.caption("Verified optimal configurations and design-space results")

required = [
    RESULTS_DIR / "optimal_config_DAC.json",
    RESULTS_DIR / "optimal_config_INDUSTRIAL.json",
    RESULTS_DIR / "training_data_DAC.csv",
    RESULTS_DIR / "training_data_INDUSTRIAL.csv",
]

missing = [p.name for p in required if not p.exists()]
if missing:
    st.error(
        "Results are missing. Run `python run_pipeline.py` first. Missing: "
        + ", ".join(missing)
    )
    st.stop()


def load_mode(mode):
    with open(RESULTS_DIR / f"optimal_config_{mode}.json") as f:
        optimal = json.load(f)
    data = pd.read_csv(RESULTS_DIR / f"training_data_{mode}.csv")
    profile = pd.read_csv(RESULTS_DIR / f"optimal_profile_{mode}.csv")
    fi = pd.read_csv(RESULTS_DIR / f"feature_importance_{mode}.csv")
    return optimal, data, profile, fi


def show_mode(mode, label):
    optimal, data, profile, fi = load_mode(mode)
    p = optimal["parameters"]
    perf = optimal["performance"]

    st.subheader(f"{label} — Optimal Verified Configuration")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Capture efficiency", f"{perf['efficiency']:.2f}%")
    c2.metric("Capture cost", f"${perf['cost']:,.2f}/t CO₂")
    c3.metric("Annual CO₂ capture", f"{perf['CO2_tpy']:,.0f} t/yr")
    c4.metric("Energy intensity", f"{perf['energy_kWh_per_t']:,.0f} kWh/t")

    config_df = pd.DataFrame(
        [
            ("Absorber diameter", p["D"], "m"),
            ("Absorber height", p["H"], "m"),
            ("Gas flow", p["G"], "m³/s"),
            ("Liquid flow", p["L"], "m³/s"),
            ("Initial NaOH concentration", p["C_NaOH0"], "mol/m³"),
            ("Total causticizer volume", p["V_total"], "m³"),
            ("CSTR stages", p["N"], "count"),
            ("Causticization rate constant", p["k_caus"], "1/s"),
            ("Equilibrium-conversion cap", p["eta_eq"], "fraction"),
        ],
        columns=["Parameter", "Value", "Unit"],
    )

    engineering_df = pd.DataFrame(
        [
            ("Absorber-only efficiency", perf["absorber_efficiency"], "%"),
            ("Causticization conversion", 100.0 * perf["regen_conversion"], "%"),
            ("Pressure drop", perf["pressure_drop_Pa"], "Pa"),
            ("Pump power", perf["pump_power_kW"], "kW"),
            ("Blower power", perf["blower_power_kW"], "kW"),
            ("Total power", perf["total_power_kW"], "kW"),
            ("Annual energy", perf["annual_energy_kWh"], "kWh/yr"),
            ("Installed cost", perf["installed_cost"], "$"),
            ("Annual total cost", perf["annual_cost"], "$/yr"),
            ("Ca(OH)₂ consumption", perf["CaOH2_tpy"], "t/yr"),
            ("CaCO₃ production", perf["CaCO3_tpy"], "t/yr"),
            ("Carbon-balance error", perf["carbon_balance_error_pct"], "%"),
            ("Superficial gas velocity", perf["superficial_gas_velocity_m_s"], "m/s"),
            ("Superficial liquid velocity", perf["superficial_liquid_velocity_m_s"], "m/s"),
            ("kLa", perf["kLa"], "1/s"),
        ],
        columns=["Metric", "Value", "Unit"],
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Optimal design variables")
        st.dataframe(config_df, hide_index=True, use_container_width=True)
    with right:
        st.markdown("#### Engineering outputs")
        st.dataframe(engineering_df, hide_index=True, use_container_width=True)

    st.markdown("### Graphs")

    fig1 = px.line(
        profile,
        x="height_m",
        y="capture_efficiency_pct",
        title=f"{label}: Capture Efficiency Along Absorber",
        labels={"height_m": "Absorber height (m)", "capture_efficiency_pct": "Capture efficiency (%)"},
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(
        data,
        x="cost",
        y="efficiency",
        color="CO2_tpy",
        title=f"{label}: Cost–Efficiency Design Space",
        labels={
            "cost": "Capture cost ($/t CO₂)",
            "efficiency": "Capture efficiency (%)",
            "CO2_tpy": "CO₂ captured (t/yr)",
        },
    )
    fig2.add_scatter(
        x=[perf["cost"]],
        y=[perf["efficiency"]],
        mode="markers",
        marker={"size": 16, "symbol": "star"},
        name="Optimal verified design",
    )
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(
        data,
        x="total_power_kW",
        y="cost",
        color="efficiency",
        title=f"{label}: Power Requirement vs Capture Cost",
        labels={
            "total_power_kW": "Total power (kW)",
            "cost": "Capture cost ($/t CO₂)",
            "efficiency": "Efficiency (%)",
        },
    )
    st.plotly_chart(fig3, use_container_width=True)

    cost_breakdown = pd.DataFrame(
        {
            "Category": ["Annualized capital", "Fixed O&M", "Electricity", "Ca(OH)₂", "Compression"],
            "Annual cost": [
                perf["annual_capital"],
                perf["fixed_OM"],
                perf["electricity_cost"],
                perf["lime_cost"],
                perf["compression_cost"],
            ],
        }
    )
    fig4 = px.bar(
        cost_breakdown,
        x="Category",
        y="Annual cost",
        title=f"{label}: Annual Cost Breakdown",
        labels={"Annual cost": "Annual cost ($/yr)"},
    )
    st.plotly_chart(fig4, use_container_width=True)

    with st.expander("Model feature importance"):
        st.dataframe(fi.sort_values("Cost importance", ascending=False), hide_index=True, use_container_width=True)

    st.markdown("### Download data")
    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "Optimal configuration CSV",
        (RESULTS_DIR / f"optimal_config_{mode}.csv").read_bytes(),
        file_name=f"optimal_config_{mode}.csv",
        mime="text/csv",
    )
    d2.download_button(
        "Optimal profile CSV",
        (RESULTS_DIR / f"optimal_profile_{mode}.csv").read_bytes(),
        file_name=f"optimal_profile_{mode}.csv",
        mime="text/csv",
    )
    d3.download_button(
        "Full design-space CSV",
        (RESULTS_DIR / f"training_data_{mode}.csv").read_bytes(),
        file_name=f"training_data_{mode}.csv",
        mime="text/csv",
    )


tab1, tab2 = st.tabs(["Direct Air Capture", "Flue Gas Capture"])
with tab1:
    show_mode("DAC", "Direct Air Capture")
with tab2:
    show_mode("INDUSTRIAL", "Flue Gas Capture")
