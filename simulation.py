import numpy as np
from scipy.integrate import solve_ivp

R = 8.314
T = 298.15
P = 101325.0
SEC_PER_YEAR = 365.0 * 24.0 * 3600.0
HOURS_PER_YEAR = 8760.0

g = 9.81
rho_liquid = 1000.0
mu_liquid = 0.001
mu_gas = 1.8e-5

H_CO2 = 3.40e4  # Pa m^3 / mol
D_CO2 = 1.9e-9  # m^2 / s
MW_CO2 = 44.01e-3  # kg / mol
MW_CaOH2 = 74.09e-3  # kg / mol
MW_CaCO3 = 100.09e-3  # kg / mol

capacity_factor = 0.85
discount_rate = 0.08
plant_life = 20
CRF = (
    discount_rate * (1.0 + discount_rate) ** plant_life
    / ((1.0 + discount_rate) ** plant_life - 1.0)
)

# The original code used 8e3 and 1e4 while concentrations were in mol/m^3.
# These values are converted here to m^3/(mol s), assuming the original
# constants were expressed in m^3/(kmol s).
K1_OH = 8.0
K2_OH = 10.0

MODE_DATA = {
    "INDUSTRIAL": {
        "yCO2": 0.12,
        "compress_cost": 18.0,
        "blower_eff": 0.72,
        "pump_eff": 0.75,
        "rho_gas": 1.45,
    },
    "DAC": {
        "yCO2": 0.00042,
        "compress_cost": 35.0,
        "blower_eff": 0.65,
        "pump_eff": 0.70,
        "rho_gas": 1.20,
    },
}


def _invalid_result(reason: str):
    return 0.0, 1e9, 0.0, {"valid": False, "invalid_reason": reason}


def run_full_model(
    mode,
    D,
    H,
    G,
    L,
    C_NaOH0,
    V_total,
    N,
    k_caus,
    eta_eq,
    yCO2=None,
    elec_price=0.10,
    lime_price=100.0,
    relative_humidity=0.50,
    T_in=298.15,
):
    """Run one physics/economic simulation.

    Units:
      D, H: m
      G, L: m^3/s
      C_NaOH0: mol/m^3
      V_total: m^3
      k_caus: 1/s
      eta_eq: fraction
      elec_price: $/kWh
      lime_price: $/metric tonne Ca(OH)2
    """
    mode = mode.upper()
    if mode not in MODE_DATA:
        raise ValueError("mode must be 'DAC' or 'INDUSTRIAL'")

    if min(D, H, G, L, C_NaOH0, V_total, k_caus, eta_eq) <= 0 or N < 1:
        return _invalid_result("Non-positive design variable")

    cfg = MODE_DATA[mode]
    if yCO2 is None:
        yCO2 = cfg["yCO2"]

    A = np.pi * (D / 2.0) ** 2
    vG = G / A
    vL = L / A

    # Keep the sampled design inside the hydrodynamic region used for this model.
    # Designs outside these ranges are not evaluated/optimized.
    if not (0.01 <= vG <= 0.35):
        return _invalid_result("Superficial gas velocity outside 0.01-0.35 m/s")
    if not (0.001 <= vL <= 0.20):
        return _invalid_result("Superficial liquid velocity outside 0.001-0.20 m/s")

    rho_g = cfg["rho_gas"]
    pump_eff = cfg["pump_eff"]
    blower_eff = cfg["blower_eff"]
    compression_cost_per_t = cfg["compress_cost"]

    # Bubble-column hydrodynamics and liquid-side transfer model.
    bubble_diameter = np.clip(
        0.0035 * (max(vG, 1e-8) / 0.1) ** (-0.15), 0.0015, 0.008
    )
    gas_holdup = np.clip(
        0.12 * (max(vG, 1e-8) / 0.1) ** 0.55, 0.01, 0.40
    )
    interfacial_area = 6.0 * gas_holdup / bubble_diameter

    Re_L = rho_liquid * vL * bubble_diameter / mu_liquid
    Sc_L = mu_liquid / (rho_liquid * D_CO2)
    Sh_L = 2.0 + 0.6 * np.sqrt(max(Re_L, 1e-12)) * Sc_L ** (1.0 / 3.0)
    kL = Sh_L * D_CO2 / bubble_diameter

    Ha = np.sqrt(max(K1_OH * C_NaOH0 * D_CO2, 1e-20)) / max(kL, 1e-12)
    if Ha < 0.3:
        enhancement_factor = 1.0
    elif Ha < 3.0:
        enhancement_factor = np.sqrt(1.0 + Ha**2)
    else:
        enhancement_factor = Ha

    kLa = kL * interfacial_area
    kLa_effective = kLa * enhancement_factor

    # Inlet gas concentration corrected for water vapour.
    T_C = T_in - 273.15
    P_sat_water = 610.94 * np.exp((17.625 * T_C) / (T_C + 243.04))
    P_H2O = np.clip(relative_humidity, 0.0, 1.0) * P_sat_water
    P_dry = max(P - P_H2O, 1.0)
    P_CO2_in = yCO2 * P_dry
    Cg0 = P_CO2_in / (R * T_in)

    def absorber(z, y):
        Cg, CO2_aq, HCO3, CO3, OH = y
        Cg = max(Cg, 0.0)
        CO2_aq = max(CO2_aq, 0.0)
        HCO3 = max(HCO3, 0.0)
        OH = max(OH, 0.0)

        P_CO2 = Cg * R * T_in
        C_star = P_CO2 / H_CO2
        transfer = kLa_effective * max(C_star - CO2_aq, 0.0)
        r1 = K1_OH * CO2_aq * OH
        r2 = K2_OH * HCO3 * OH

        dCg = -transfer / max(vG, 1e-12)
        dCO2_aq = (transfer - r1) / max(vL, 1e-12)
        dHCO3 = (r1 - r2) / max(vL, 1e-12)
        dCO3 = r2 / max(vL, 1e-12)
        dOH = (-r1 - r2) / max(vL, 1e-12)
        return [dCg, dCO2_aq, dHCO3, dCO3, dOH]

    z_eval = np.linspace(0.0, H, 160)
    try:
        absorber_sol = solve_ivp(
            absorber,
            [0.0, H],
            [Cg0, 0.0, 0.0, 0.0, C_NaOH0],
            method="BDF",
            t_eval=z_eval,
            rtol=1e-6,
            atol=1e-9,
        )
        if not absorber_sol.success or absorber_sol.y.shape[1] != len(z_eval):
            return _invalid_result("Absorber integration failed")
    except Exception as exc:
        return _invalid_result(f"Absorber integration error: {exc}")

    Cg_profile = np.maximum(absorber_sol.y[0], 0.0)
    CO2_aq_profile = np.maximum(absorber_sol.y[1], 0.0)
    HCO3_profile = np.maximum(absorber_sol.y[2], 0.0)
    CO3_profile = np.maximum(absorber_sol.y[3], 0.0)
    OH_profile = np.maximum(absorber_sol.y[4], 0.0)

    Cg_out = Cg_profile[-1]
    inlet_CO2_mol_s = G * Cg0
    absorber_CO2_mol_s = max(G * (Cg0 - Cg_out), 0.0)
    absorber_efficiency = float(
        np.clip(100.0 * absorber_CO2_mol_s / max(inlet_CO2_mol_s, 1e-12), 0.0, 100.0)
    )

    # Steady-state first-order CSTR train approximation for causticization.
    reactor_volume = V_total / max(N, 1)
    tau_stage = reactor_volume / max(L, 1e-12)
    kinetic_conversion = 1.0 - (
        1.0 / (1.0 + max(k_caus, 0.0) * tau_stage)
    ) ** max(N, 1)
    regen_conversion = float(np.clip(min(kinetic_conversion, eta_eq), 0.0, 1.0))

    # Sustainable loop capture is limited by the fraction successfully causticized.
    captured_CO2_mol_s = absorber_CO2_mol_s * regen_conversion
    efficiency = absorber_efficiency * regen_conversion

    # kg/mol * mol/s * s/year -> kg/year; /1000 -> metric tonnes/year.
    CO2_tpy = (
        captured_CO2_mol_s
        * MW_CO2
        * SEC_PER_YEAR
        * capacity_factor
        / 1000.0
    )
    if CO2_tpy <= 1e-9:
        return _invalid_result("Negligible capture")

    # Carbon conservation check across the absorber only.
    carbon_in_mol_s = G * Cg0
    carbon_out_mol_s = G * Cg_out + L * (
        CO2_aq_profile[-1] + HCO3_profile[-1] + CO3_profile[-1]
    )
    carbon_balance_error_pct = (
        abs(carbon_in_mol_s - carbon_out_mol_s)
        / max(carbon_in_mol_s, 1e-12)
        * 100.0
    )

    # Pressure drop and power.
    friction_factor = 0.02
    hydrostatic_dp = rho_liquid * (1.0 - gas_holdup) * g * H
    friction_dp = (
        friction_factor * (H / max(D, 0.1)) * 0.5 * rho_g * vG**2
    )
    distributor_dp = 1500.0
    deltaP = hydrostatic_dp + friction_dp + distributor_dp

    pump_head = H + 5.0
    pump_power_kW = rho_liquid * g * L * pump_head / pump_eff / 1000.0
    blower_power_kW = deltaP * G / blower_eff / 1000.0
    total_power_kW = pump_power_kW + blower_power_kW
    annual_energy_kWh = total_power_kW * HOURS_PER_YEAR * capacity_factor
    electricity_cost = annual_energy_kWh * elec_price

    # Equipment and annualized economics.
    absorber_volume = A * H
    absorber_cost = 18000.0 * absorber_volume**0.62
    causticizer_cost = 22000.0 * V_total**0.60
    pump_cost = 8000.0 * max(L, 0.1) ** 0.70
    blower_cost = 12000.0 * max(G, 0.1) ** 0.72
    installed_cost = (
        absorber_cost + causticizer_cost + pump_cost + blower_cost
    ) * 3.2 * 1.15

    # 5% Ca(OH)2 excess is used consistently with the causticization assumption.
    CaOH2_tpy = 1.05 * CO2_tpy * (MW_CaOH2 / MW_CO2)
    CaCO3_tpy = CO2_tpy * (MW_CaCO3 / MW_CO2)
    lime_cost_total = CaOH2_tpy * lime_price

    fixed_OM = 0.05 * installed_cost
    compression_cost = compression_cost_per_t * CO2_tpy
    annual_capital = installed_cost * CRF
    annual_cost = (
        annual_capital
        + fixed_OM
        + electricity_cost
        + lime_cost_total
        + compression_cost
    )
    cost_per_t = annual_cost / CO2_tpy

    capture_eff_profile = (
        100.0 * (1.0 - Cg_profile / max(Cg_profile[0], 1e-12)) * regen_conversion
    )

    results = {
        "valid": True,
        "mode": mode,
        "yCO2": float(yCO2),
        "CO2_tpy": float(CO2_tpy),
        "efficiency": float(efficiency),
        "absorber_efficiency": float(absorber_efficiency),
        "regen_conversion": float(regen_conversion),
        "cost_per_t": float(cost_per_t),
        "pressure_drop_Pa": float(deltaP),
        "pump_power_kW": float(pump_power_kW),
        "blower_power_kW": float(blower_power_kW),
        "total_power_kW": float(total_power_kW),
        "annual_energy_kWh": float(annual_energy_kWh),
        "energy_kWh_per_t": float(annual_energy_kWh / CO2_tpy),
        "absorber_cost": float(absorber_cost),
        "causticizer_cost": float(causticizer_cost),
        "pump_cost": float(pump_cost),
        "blower_cost": float(blower_cost),
        "installed_cost": float(installed_cost),
        "annual_capital": float(annual_capital),
        "fixed_OM": float(fixed_OM),
        "compression_cost": float(compression_cost),
        "lime_cost": float(lime_cost_total),
        "electricity_cost": float(electricity_cost),
        "annual_cost": float(annual_cost),
        "CaOH2_tpy": float(CaOH2_tpy),
        "CaCO3_tpy": float(CaCO3_tpy),
        "bubble_diameter_m": float(bubble_diameter),
        "gas_holdup": float(gas_holdup),
        "kLa": float(kLa),
        "kLa_effective": float(kLa_effective),
        "superficial_gas_velocity_m_s": float(vG),
        "superficial_liquid_velocity_m_s": float(vL),
        "carbon_balance_error_pct": float(carbon_balance_error_pct),
        "height_profile": z_eval,
        "CO2_profile": Cg_profile,
        "capture_efficiency_profile": capture_eff_profile,
        "CO2_aq_profile": CO2_aq_profile,
        "HCO3_profile": HCO3_profile,
        "CO3_profile": CO3_profile,
        "OH_profile": OH_profile,
    }

    return CO2_tpy, cost_per_t, efficiency, results
