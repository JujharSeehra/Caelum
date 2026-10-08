from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.optimize import differential_evolution

from simulation import run_full_model

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

NUM_SAMPLES = 3000
RANDOM_SEED = 42
OPT_MAXITER = 40

FEATURES = [
    "D", "H", "G", "L", "C_NaOH0", "V_total", "N", "k_caus", "eta_eq"
]

MIN_EFFICIENCY = {"DAC": 70.0, "INDUSTRIAL": 90.0}

RANGES = {
    "DAC": {
        "D": (15.0, 35.0),
        "H": (2.0, 15.0),
        "G": (20.0, 100.0),
        "L": (2.0, 15.0),
        "C_NaOH0": (500.0, 3000.0),
        "V_total": (50.0, 300.0),
        "N": (2, 6),
        "k_caus": (0.05, 0.50),
        "eta_eq": (0.85, 0.99),
    },
    "INDUSTRIAL": {
        "D": (2.0, 10.0),
        "H": (2.0, 20.0),
        "G": (0.5, 5.0),
        "L": (0.5, 5.0),
        "C_NaOH0": (500.0, 3000.0),
        "V_total": (20.0, 200.0),
        "N": (2, 5),
        "k_caus": (0.05, 0.50),
        "eta_eq": (0.85, 0.97),
    },
}


def sample_parameters(rng, mode):
    r = RANGES[mode]
    return {
        "D": rng.uniform(*r["D"]),
        "H": rng.uniform(*r["H"]),
        "G": rng.uniform(*r["G"]),
        "L": rng.uniform(*r["L"]),
        "C_NaOH0": rng.uniform(*r["C_NaOH0"]),
        "V_total": rng.uniform(*r["V_total"]),
        "N": int(rng.integers(r["N"][0], r["N"][1] + 1)),
        "k_caus": rng.uniform(*r["k_caus"]),
        "eta_eq": rng.uniform(*r["eta_eq"]),
    }


def simulate_row(mode, p):
    CO2_tpy, cost, efficiency, res = run_full_model(mode=mode, **p)
    if not res.get("valid", False):
        return None
    if not all(np.isfinite(x) for x in [CO2_tpy, cost, efficiency]):
        return None

    return {
        **p,
        "CO2_tpy": CO2_tpy,
        "cost": cost,
        "efficiency": efficiency,
        "absorber_efficiency": res["absorber_efficiency"],
        "regen_conversion": res["regen_conversion"],
        "pressure_drop_Pa": res["pressure_drop_Pa"],
        "pump_power_kW": res["pump_power_kW"],
        "blower_power_kW": res["blower_power_kW"],
        "total_power_kW": res["total_power_kW"],
        "annual_energy_kWh": res["annual_energy_kWh"],
        "energy_kWh_per_t": res["energy_kWh_per_t"],
        "installed_cost": res["installed_cost"],
        "annual_cost": res["annual_cost"],
        "annual_capital": res["annual_capital"],
        "fixed_OM": res["fixed_OM"],
        "electricity_cost": res["electricity_cost"],
        "lime_cost": res["lime_cost"],
        "compression_cost": res["compression_cost"],
        "CaOH2_tpy": res["CaOH2_tpy"],
        "CaCO3_tpy": res["CaCO3_tpy"],
        "gas_holdup": res["gas_holdup"],
        "kLa": res["kLa"],
        "kLa_effective": res["kLa_effective"],
        "superficial_gas_velocity_m_s": res["superficial_gas_velocity_m_s"],
        "superficial_liquid_velocity_m_s": res["superficial_liquid_velocity_m_s"],
        "carbon_balance_error_pct": res["carbon_balance_error_pct"],
    }


def generate_dataset(mode):
    rng = np.random.default_rng(RANDOM_SEED + (0 if mode == "DAC" else 1))
    rows = []
    attempts = 0
    max_attempts = NUM_SAMPLES * 8

    while len(rows) < NUM_SAMPLES and attempts < max_attempts:
        attempts += 1
        p = sample_parameters(rng, mode)
        row = simulate_row(mode, p)
        if row is not None:
            rows.append(row)

        if len(rows) and len(rows) % 500 == 0 and len(rows) != getattr(generate_dataset, "_last", None):
            print(f"{mode}: {len(rows):,}/{NUM_SAMPLES:,} valid simulations")
            generate_dataset._last = len(rows)

    if len(rows) < NUM_SAMPLES:
        raise RuntimeError(
            f"Could only generate {len(rows)} valid {mode} simulations after {attempts} attempts."
        )

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / f"training_data_{mode}.csv"
    df.to_csv(out, index=False)
    return df


def train_models(mode, df):
    X = df[FEATURES]
    yc = df["cost"]
    ye = df["efficiency"]

    X_train, X_test, yc_train, yc_test, ye_train, ye_test = train_test_split(
        X, yc, ye, test_size=0.20, random_state=RANDOM_SEED
    )

    cost_model = RandomForestRegressor(
        n_estimators=150, random_state=RANDOM_SEED, n_jobs=-1
    )
    eff_model = RandomForestRegressor(
        n_estimators=150, random_state=RANDOM_SEED, n_jobs=-1
    )
    cost_model.fit(X_train, yc_train)
    eff_model.fit(X_train, ye_train)

    cp = cost_model.predict(X_test)
    ep = eff_model.predict(X_test)

    metrics = {
        "mode": mode,
        "rows": int(len(df)),
        "cost_r2": float(r2_score(yc_test, cp)),
        "cost_mae": float(mean_absolute_error(yc_test, cp)),
        "efficiency_r2": float(r2_score(ye_test, ep)),
        "efficiency_mae": float(mean_absolute_error(ye_test, ep)),
    }

    joblib.dump(cost_model, MODELS_DIR / f"cost_model_{mode}.pkl")
    joblib.dump(eff_model, MODELS_DIR / f"efficiency_model_{mode}.pkl")

    fi = pd.DataFrame({
        "Feature": FEATURES,
        "Cost importance": cost_model.feature_importances_,
        "Efficiency importance": eff_model.feature_importances_,
    })
    fi.to_csv(RESULTS_DIR / f"feature_importance_{mode}.csv", index=False)

    return cost_model, eff_model, metrics


def make_bounds(mode):
    r = RANGES[mode]
    return [r["D"],r["H"],r["G"],r["L"],r["C_NaOH0"],r["V_total"],r["N"],r["k_caus"],r["eta_eq"],]


def vector_to_params(x):
    p = dict(zip(FEATURES, x))
    p["N"] = int(round(p["N"]))
    return p


def optimize(mode, df, cost_model, eff_model):
    bounds = make_bounds(mode)
    min_eff = MIN_EFFICIENCY[mode]
    def objective(x):
        params = vector_to_params(x)
        row = simulate_row(mode, params)
        if row is None:
            return 1e12
        if row["efficiency"] < min_eff:
            return row["cost"] + 1e5 * (min_eff - row["efficiency"]) ** 2
        return row["cost"]

    opt = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=OPT_MAXITER,
        popsize=6,
        tol=1e-3,
        seed=RANDOM_SEED,
        polish=False,
        updating="immediate",
        workers=1,
    )

    optimized_candidate = vector_to_params(opt.x)
    optimized_row = simulate_row(mode, optimized_candidate)

    feasible = df[df["efficiency"] >= min_eff].copy()
    if feasible.empty:
        raise RuntimeError(
            f"No sampled {mode} configurations meet the {min_eff:.1f}% efficiency constraint."
        )
    best_sample = feasible.loc[feasible["cost"].idxmin()]
    sampled_candidate = {k: best_sample[k] for k in FEATURES}
    sampled_candidate["N"] = int(round(sampled_candidate["N"]))
    sampled_row = simulate_row(mode, sampled_candidate)

    verified_candidates = [r for r in [optimized_row, sampled_row] if r is not None]
    verified_candidates = [r for r in verified_candidates if r["efficiency"] >= min_eff]
    if not verified_candidates:
        verified_candidates = [sampled_row]

    final_row = min(verified_candidates, key=lambda r: r["cost"])
    params = {k: final_row[k] for k in FEATURES}
    params["N"] = int(round(params["N"]))

    _, _, _, full = run_full_model(mode=mode, **params)

    json_payload = {
        "mode": mode,
        "minimum_efficiency_constraint": min_eff,
        "optimization_method": "Differential evolution on the physics/economic model; RF models retained for diagnostics",
        "parameters": {k: (int(v) if k == "N" else float(v)) for k, v in params.items()},
        "performance": {
            k: float(v)
            for k, v in final_row.items()
            if k not in FEATURES and np.isscalar(v)
        },
        "profile": {
            "height_m": full["height_profile"].tolist(),
            "capture_efficiency_pct": full["capture_efficiency_profile"].tolist(),
            "gas_CO2_mol_m3": full["CO2_profile"].tolist(),
        },
    }

    with open(RESULTS_DIR / f"optimal_config_{mode}.json", "w") as f:
        json.dump(json_payload, f, indent=2)

    pd.DataFrame([final_row]).to_csv(
        RESULTS_DIR / f"optimal_config_{mode}.csv", index=False
    )
    pd.DataFrame(json_payload["profile"]).to_csv(
        RESULTS_DIR / f"optimal_profile_{mode}.csv", index=False
    )

    return final_row


def main():
    all_metrics = []
    summaries = []

    for mode in ["DAC", "INDUSTRIAL"]:
        print(f"\n=== {mode} ===")
        df = generate_dataset(mode)
        cost_model, eff_model, metrics = train_models(mode, df)
        all_metrics.append(metrics)
        best = optimize(mode, df, cost_model, eff_model)

        summaries.append({"mode": mode, **best})
        print(
            f"Optimal verified configuration: "
            f"efficiency={best['efficiency']:.2f}%, "
            f"cost=${best['cost']:.2f}/tCO2, "
            f"capture={best['CO2_tpy']:.0f} t/yr"
        )

    pd.DataFrame(all_metrics).to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
    pd.DataFrame(summaries).to_csv(RESULTS_DIR / "optimal_summary.csv", index=False)

    print("\nFinished. Run: streamlit run MLapp.py")


if __name__ == "__main__":
    main()
