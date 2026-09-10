
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from simulation import run_full_model

st.set_page_config(page_title="AI Carbon Capture Optimizer", page_icon="🏭", layout="wide")

st.title("AI Carbon Capture Optimizer")
st.markdown("""This dashboard combines:
- Machine Learning prediction
- First-principles process simulation
- Economic analysis
- Bayesian optimized design""")

mode = st.sidebar.selectbox("Capture Mode", ["DAC", "INDUSTRIAL"])
st.info(f"Current Simulation Mode: {mode}")

@st.cache_resource
def load_models(mode):
    cost_model = joblib.load(f"cost_model_{mode}.pkl")
    eff_model = joblib.load(f"efficiency_model_{mode}.pkl")
    return cost_model, eff_model

cost_model, eff_model = load_models(mode)
yCO2 = 0.12
st.sidebar.title("Design Variables")

if mode == "DAC":
    D_min, D_max, D_default = 5.0, 15.0, 8.0
    H_min, H_max, H_default = 25.0, 60.0, 40.0
    G_min, G_max, G_default = 30.0, 150.0, 80.0
    L_min, L_max, L_default = 5.0, 30.0, 12.0
    C_min, C_max, C_default = 1500, 6000, 3000
    V_min, V_max, V_default = 100, 500, 250
    N_min, N_max, N_default = 3, 6, 4
    yCO2 = 0.00042

else:
    D_min, D_max, D_default = 2.0, 8.0, 3.0
    H_min, H_max, H_default = 10.0, 35.0, 18.0
    G_min, G_max, G_default = 0.5, 5.0, 1.5
    L_min, L_max, L_default = 0.5, 5.0, 2.0
    C_min, C_max, C_default = 500, 3000, 1500
    V_min, V_max, V_default = 20, 200, 80
    N_min, N_max, N_default = 2, 5, 3
    yCO2 = 0.12

D = st.sidebar.slider("Absorber Diameter (m)", D_min, D_max, D_default, 0.1)
H = st.sidebar.slider("Absorber Height (m)", H_min, H_max, H_default, 0.5)
G = st.sidebar.slider("Gas Flow (m³/s)", G_min, G_max, G_default, 0.05)
L = st.sidebar.slider("Liquid Flow (m³/s)",L_min, L_max, L_default, 0.05)

C_NaOH0 = st.sidebar.slider("NaOH Concentration (mol/m³)", C_min, C_max, C_default, 50)


V_total = st.sidebar.slider("Causticizer Volume (m³)", V_min, V_max, V_default, 5)


N = st.sidebar.slider("Number of CSTRs", N_min, N_max, N_default)


k_caus = st.sidebar.slider("Reaction Rate Constant", 0.01, 1.00, 0.30, 0.01)


eta_eq = st.sidebar.slider("Equilibrium Conversion", 0.80, 0.99, 0.90, 0.01)


st.sidebar.divider()


verify = st.sidebar.checkbox("Run Physics Verification", value=True)


features = pd.DataFrame({"D":[D], "H":[H], "G":[G], "L":[L], "C_NaOH0":[C_NaOH0], "V_total":[V_total], "N":[N], "k_caus":[k_caus], "eta_eq":[eta_eq]})


ml_cost = float(cost_model.predict(features)[0])

ml_eff = float(eff_model.predict(features)[0])

st.header("Machine Learning Prediction")

metric1, metric2 = st.columns(2)

with metric1:

    st.metric("Predicted Capture Cost", f"${ml_cost:,.2f}/tCO₂")

with metric2:

    st.metric("Predicted Efficiency", f"{ml_eff:.2f}%")


physics = None

if verify:

    with st.spinner("Running physics simulation..."):

        CO2_tpy, sim_cost, sim_eff, physics = run_full_model(    mode = mode, D=D,H=H,G=G,L=L,C_NaOH0=C_NaOH0,V_total=V_total,N=N,k_caus=k_caus,eta_eq=eta_eq)

if physics is not None:

    st.header("⚙️ Physics Simulation")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric("CO₂ Captured", f"{CO2_tpy:,.0f} t/year")

    with c2:

        st.metric("Simulation Cost", f"${sim_cost:,.2f}/tCO₂")

    with c3:

        st.metric("Simulation Efficiency", f"{sim_eff:.2f}%")

if physics is not None:

    st.header("ML vs Physics")

    compare = pd.DataFrame({

        "Model":[

            "Machine Learning",

            "Physics"

        ],

        "Cost":[

            ml_cost,

            sim_cost

        ],

        "Efficiency":[

            ml_eff,

            sim_eff

        ]

    })

    st.dataframe(compare, use_container_width=True)

    cost_error = abs(ml_cost-sim_cost)

    eff_error = abs(ml_eff-sim_eff)

    st.success(f"Cost Error: ${cost_error:.2f}/t    |    Efficiency Error: {eff_error:.2f}%")

st.divider()

st.header("Process Visualizations")

cost_chart = st.container()

energy_chart = st.container()

power_chart = st.container()

download_section = st.container()

if physics is not None:

    with cost_chart:

        st.subheader("💰 Annual Cost Breakdown")

        cost_df = pd.DataFrame({

            "Category":["Capital Recovery", "Fixed O&M", "Electricity", "Lime", "Compression"],

            "Cost ($/yr)":[physics["installed_cost"]*0.1019, physics["installed_cost"]*0.05, physics["electricity_cost"], physics["lime_cost"], 35*physics["CO2_tpy"]]

        })

        fig = px.bar(

            cost_df,

            x="Category",

            y="Cost ($/yr)",

            title="Annual Operating Cost Breakdown")

        fig.update_layout(height=450)

        st.plotly_chart(fig, use_container_width=True)

if physics is not None:

    st.subheader("🥧 Operating Cost Distribution")

    pie = px.pie(

        cost_df,

        values="Cost ($/yr)",

        names="Category",

        hole=0.45)

    pie.update_layout(height=500)

    st.plotly_chart(    pie, use_container_width=True)

if physics is not None:

    with power_chart:

        st.subheader("⚡ Power Consumption")

        power_df = pd.DataFrame({

            "Equipment":["Pump", "Blower"],

            "Power (kW)":[physics["pump_power_kW"], physics["blower_power_kW"]]

        })

        fig2 = px.bar(

            power_df,

            x="Equipment",

            y="Power (kW)",

            text="Power (kW)",

            title="Equipment Power Consumption")

        fig2.update_layout(height=450)

        st.plotly_chart(fig2, use_container_width=True)

if physics is not None:

    st.header("Engineering Summary")

    summary = pd.DataFrame({

        "Metric":[

            "Annual CO₂ Capture",

            "Capture Efficiency",

            "Capture Cost",

            "Pressure Drop",

            "Pump Power",

            "Blower Power",

            "Total Power",

            "Annual Energy",

            "Installed Cost",

            "Annual Operating Cost"

        ],

        "Value":[

            f"{CO2_tpy:,.0f} t/year",

            f"{physics['efficiency']:.2f} %",

            f"${physics['cost_per_t']:.2f}/tCO₂",

            f"{physics['pressure_drop_Pa']:,.0f} Pa",

            f"{physics['pump_power_kW']:.2f} kW",

            f"{physics['blower_power_kW']:.2f} kW",

            f"{physics['total_power_kW']:.2f} kW",

            f"{physics['annual_energy_kWh']:,.0f} kWh",

            f"${physics['installed_cost']:,.0f}",

            f"${physics['annual_cost']:,.0f}"

        ]

    })

    st.dataframe(summary,hide_index=True, use_container_width=True)


if physics is not None:

    st.header("Design Diagnostics")

    if physics["efficiency"] < 75:

        st.warning("Capture efficiency is below 55%. Consider increasing absorber height or NaOH concentration.")

    elif physics["efficiency"] > 98:

        st.success("Excellent capture efficiency.")

    if physics["pressure_drop_Pa"] > 10000:

        st.error("Pressure drop is very high. Operating costs may increase significantly.")

    elif physics["pressure_drop_Pa"] > 5000:

        st.warning("Moderately high pressure drop.")

    else:

        st.success("Pressure drop is within a reasonable operating range.")


if physics is not None:

    with download_section:

        st.header("Export Results")

        export = pd.DataFrame({

            "Parameter":["Diameter","Height","Gas Flow","Liquid Flow","NaOH","Volume","CSTRs","Reaction Rate","Efficiency","CO₂ Captured","Capture Cost","Installed Cost","Pressure Drop","Total Power"
],

            "Value":[D,H,G,L,C_NaOH0,V_total,N,k_caus,physics["efficiency"],physics["CO2_tpy"],physics["cost_per_t"],physics["installed_cost"],physics["pressure_drop_Pa"],physics["total_power_kW"]
]

        })

        csv = export.to_csv(index=False)

        st.download_button(

            "Download Results (.csv)",

            csv,

            file_name="simulation_results.csv",

            mime="text/csv")

st.divider()

st.caption("AI Carbon Capture Optimizer • Physics-Based Simulation + Machine Learning • Developed using Python, SciPy, Scikit-Learn and Streamlit")

st.divider()

st.header("🔬 Process Behavior Visualization")


if mode=="DAC":
    yCO2=0.00042
else:
    yCO2=0.12 
def absorber_profile(D, H, G, L, C_NaOH0, yCO2):

    import numpy as np
    from scipy.integrate import solve_ivp  
    R = 8.314
    T = 298
    P = 101325

    A = np.pi*(D/2)**2

    vG = G/A
    vL = L/A

    kLa = 0.28*(vG/0.1)**0.7

    k_rxn = 8000
    H_CO2 = 3.4e4

    Cg0 = yCO2*P/(R*T)


    def model(z,y):

        Cg,Cl,NaOH=y

        P_CO2=Cg*R*T

        C_star=P_CO2/H_CO2

        transfer=0.9*kLa*(C_star-Cl)

        reaction=k_rxn*Cl*(NaOH/(NaOH+1000))

        return [
            -transfer/vG, (transfer-reaction)/vL, -2*reaction/vL
        ]


    z=np.linspace(0,H,200)

    sol=solve_ivp(    model, [0,H], [Cg0,0,C_NaOH0], t_eval=z)

    return z, sol.y[0], sol.y[2], Cg0



z,Cg,NaOH,Cg0 = absorber_profile(D, H, G, L, C_NaOH0, yCO2)


profile_df=pd.DataFrame({

    "Height (m)":z,

    "CO2 Gas Concentration":Cg,

    "NaOH Concentration":NaOH

})


# CO2 concentration

fig1=px.line(profile_df, x="Height (m)", y="CO2 Gas Concentration", title="CO₂ Concentration Through Absorber Height")

st.plotly_chart(fig1, use_container_width=True)


# NaOH concentration

fig2=px.line(profile_df, x="Height (m)", y="NaOH Concentration", title="NaOH Consumption Through Absorber")

st.plotly_chart(fig2, use_container_width=True)


profile_df["Capture Efficiency (%)"] = (100*(1-profile_df["CO2 Gas Concentration"]/Cg0))


fig3=px.line(profile_df, x="Height (m)", y="Capture Efficiency (%)", title="CO₂ Capture Efficiency Along Column")

st.plotly_chart(fig3, use_container_width=True)


st.header("Design Space Analysis")


try:

    data=pd.read_csv(    f"training_data_{mode}.csv")


    fig4=px.scatter(    data, x="cost", y="efficiency", color="CO2_tpy", title="Cost vs Efficiency Tradeoff", labels={
            "cost":"Capture Cost ($/tCO₂)", "efficiency":"Efficiency (%)"
        })


    st.plotly_chart(    fig4, use_container_width=True)


    fig5=px.scatter(    data, x="total_power_kW", y="cost", color="efficiency", title="Energy Consumption vs Cost")


    st.plotly_chart(    fig5, use_container_width=True)


    fig6=px.scatter(    data, x="D", y="cost", color="efficiency", title="Absorber Diameter Effect on Cost")


    st.plotly_chart(    fig6, use_container_width=True)


except Exception:

    st.warning(    "training_data.csv not found. Run data_generation.py first.")