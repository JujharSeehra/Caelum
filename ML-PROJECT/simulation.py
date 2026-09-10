import numpy as np
from scipy.integrate import solve_ivp

MODE = "DAC"

R = 8.314
T = 298.15
P = 101325

SEC_PER_YEAR = 365 * 24 * 3600

g = 9.81

rho_liquid = 1000
rho_gas = 1.20 

mu_liquid = 0.001

H_CO2 = 3.40e4

capacity_factor = 0.85

discount_rate = 0.08
plant_life = 20

CRF = (discount_rate * (1 + discount_rate) ** plant_life) / ((1 + discount_rate) ** plant_life - 1)

MODE_DATA = {
    "INDUSTRIAL":{
        "yCO2":0.12,
        "default_D":3.0,
        "default_H":18.0,
        "default_G":1.5,
        "default_L":2.0,
        "default_C":1500,
        "default_V":80,
        "default_N":3,
        "default_k":0.30,
        "default_eta":0.92,
        "compress_cost":18,
        "blower_eff":0.72,
        "pump_eff":0.75,
        "rho_gas":1.45

    },
    "DAC":{
        "yCO2":0.00042,
        "default_D":8.0,
        "default_H":35.0,
        "default_G":80.0,
        "default_L":12.0,
        "default_C":3000,
        "default_V":250,
        "default_N":4,
        "default_k":0.45,
        "default_eta":0.95,
        "compress_cost":35,
        "blower_eff":0.65,
        "pump_eff":0.70,
        "rho_gas":1.20
    }
}
def run_full_model(mode, D=None, H=None, G=None, L=None, yCO2=None, C_NaOH0=None, V_total=None, N=None, k_caus=None, eta_eq=None, elec_price=0.10, lime_price=100):
    mode = mode.upper()
    if mode not in MODE_DATA:
        raise ValueError("Mode must be 'DAC' or 'INDUSTRIAL'.")
    defaults = MODE_DATA[mode]
    if D is None:
        D = defaults["default_D"]
    if H is None:
        H = defaults["default_H"]
    if G is None:
        G = defaults["default_G"]
    if L is None:
        L = defaults["default_L"]
    if yCO2 is None:
        yCO2 = defaults["yCO2"]
    if C_NaOH0 is None:
        C_NaOH0 = defaults["default_C"]
    if V_total is None:
        V_total = defaults["default_V"]
    if N is None:
        N = defaults["default_N"]
    if k_caus is None:
        k_caus = defaults["default_k"]
    if eta_eq is None:
        eta_eq = defaults["default_eta"]
    rho_g = defaults["rho_gas"]
    pump_eff = defaults["pump_eff"]
    blower_eff = defaults["blower_eff"]
    compression_cost_per_t = defaults["compress_cost"]
    A = np.pi * (D / 2) ** 2

    if A <= 0:
        return 0, 1e9, 0, {}
    vG = G / A
    vL = L / A

    bubble_diameter = 0.004 + 0.0015 * np.sqrt(max(vG,1e-6))

    gas_holdup = min(0.35, 0.08 * (max(vG,1e-6)/0.1)**0.6)

    interfacial_area = (6 * gas_holdup / bubble_diameter)

    kL = 1.5e-3 * (max(vL, 1e-6) / 0.05) ** 0.35

    kLa = kL * interfacial_area

    k_rxn = 8000

    Cg0 = yCO2 * P / (R * T)

    Cl0 = 0.0

    def absorber(z,y):
        Cg,Cl,NaOH = y
        P_CO2 = Cg * R * T
        C_star = P_CO2 / H_CO2
        transfer = kLa * max(C_star - Cl,0)
        reaction = (k_rxn * max(Cl,0) * (max(NaOH,0)/(max(NaOH,0)+1000)))
        dCg = -transfer / max(vG,1e-8)
        dCl = (transfer-reaction)/max(vL,1e-8)
        dNaOH = -2*reaction/max(vL,1e-8)
        return [dCg, dCl, dNaOH]

    try:
        z_eval = np.linspace(0,H,200)
        absorber_sol = solve_ivp(absorber, [0,H], [Cg0,Cl0,C_NaOH0],method="RK45", t_eval=z_eval)

    except Exception:
        return 0,1e9,0,{}

    Cg_profile = absorber_sol.y[0]
    Cl_profile = absorber_sol.y[1]
    NaOH_profile = absorber_sol.y[2]
    Cg_out = max(Cg_profile[-1],0)
    CO2_abs_mol_s = max(G * (Cg0-Cg_out), 0)
    efficiency = min(100 * CO2_abs_mol_s / max(G*Cg0,1e-12),100)
    reactor_volume = V_total / max(N,1)
    tau = reactor_volume / max(L,1e-8)
    tspan = np.linspace(0, 4*tau, 150)
    Na2CO3 = CO2_abs_mol_s
    NaOH = 0.0
    CaOH2 = 1.05 * Na2CO3

    for stage in range(N):
        def cstr(t,y):
            Na2CO3,NaOH,CaOH2 = y
            rate = min(k_caus * Na2CO3, eta_eq * Na2CO3 / max(tau,1e-8))
            return [-rate, 2*rate,-rate]

        cstr_sol = solve_ivp(cstr, [0,tspan[-1]], [Na2CO3,NaOH,CaOH2],t_eval=tspan)

        Na2CO3 = cstr_sol.y[0,-1]
        NaOH = cstr_sol.y[1,-1]
        CaOH2 = cstr_sol.y[2,-1]
    CO2_tpy = (CO2_abs_mol_s * 44.01/1000 * SEC_PER_YEAR * capacity_factor)

    if CO2_tpy <= 1e-6:
        return 0,1e9,0,{}
    friction_factor = 0.02

    hydrostatic_dp = rho_g * g * H

    friction_dp = (friction_factor * (H / max(D, 0.1)) * 0.5 * rho_g * vG**2)
    distributor_dp = 1500
    deltaP = (hydrostatic_dp + friction_dp + distributor_dp)
    pump_head = H + 5
    pump_power = (rho_liquid * g * L * pump_head / pump_eff)

    blower_power = (deltaP * G / blower_eff)

    total_power = (pump_power + blower_power)

    total_power_kW = total_power / 1000
    annual_energy = (total_power_kW * 8760 * capacity_factor)

    electricity_cost = (annual_energy* elec_price)

    absorber_volume = A * H

    absorber_cost = (18000 * absorber_volume**0.62)

    causticizer_cost = (22000 * V_total**0.60)

    pump_cost = (8000 * max(L,0.1)**0.70)

    blower_cost = (12000* max(G,0.1)**0.72)

    installed_cost = (absorber_cost + causticizer_cost + pump_cost + blower_cost) * 3.2 * 1.15

    CaOH2_tpy = (CO2_tpy * (74.09 / 44.01))

    lime_cost_total = (CaOH2_tpy * lime_price)

    fixed_OM = (0.05 *installed_cost)

    compression_cost = (compression_cost_per_t * CO2_tpy)

    annual_capital = (installed_cost * CRF)

    annual_cost = (annual_capital + fixed_OM + electricity_cost + lime_cost_total + compression_cost)

    cost_per_t = annual_cost / CO2_tpy

    results = {

        "mode": mode,
        "yCO2": yCO2,
        "CO2_tpy": CO2_tpy,
        "efficiency": efficiency,
        "cost_per_t": cost_per_t,
        "pressure_drop_Pa": deltaP,
        "pump_power_kW": pump_power / 1000,
        "blower_power_kW": blower_power / 1000,
        "total_power_kW": total_power_kW,
        "annual_energy_kWh": annual_energy,
        "absorber_cost": absorber_cost,
        "causticizer_cost": causticizer_cost,
        "pump_cost": pump_cost,
        "blower_cost": blower_cost,
        "installed_cost": installed_cost,
        "annual_capital": annual_capital,
        "fixed_OM": fixed_OM,
        "compression_cost": compression_cost,
        "lime_cost": lime_cost_total,
        "electricity_cost": electricity_cost,
        "annual_cost": annual_cost,
        "bubble_diameter_m": bubble_diameter,
        "gas_holdup": gas_holdup,
        "kLa": kLa,
        "CO2_profile": Cg_profile,
        "NaOH_profile": NaOH_profile,
        "height_profile": z_eval

    }

    return (CO2_tpy, cost_per_t,efficiency, results)

if __name__ == "__main__":

    print("=" * 60)

    print("DIRECT AIR CAPTURE")

    print("=" * 60)

    CO2_tpy, cost, eff, results = run_full_model(mode="DAC")

    print(f"CO₂ Captured: {CO2_tpy:,.0f} t/year")
    print(f"Capture Cost: ${cost:,.2f}/tCO₂")
    print(f"Single-Pass Efficiency: {eff:.2f}%")

    print()

    print("INDUSTRIAL FLUE GAS")
    print()

    CO2_tpy, cost, eff, results = run_full_model(mode="INDUSTRIAL")

    print(f"CO₂ Captured: {CO2_tpy:,.0f} t/year")
    print(f"Capture Cost: ${cost:,.2f}/tCO₂")
    print(f"Single-Pass Efficiency: {eff:.2f}%")