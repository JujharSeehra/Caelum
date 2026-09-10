import numpy as np
import pandas as pd
import joblib

from simulation import run_full_model

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args


MODES = [
    "DAC",
    "INDUSTRIAL"
]


for mode in MODES:

    print("\n")
    print(f"TRAINING {mode}")
    print("=" * 60)

    filename = f"training_data_{mode}.csv"

    print(f"Loading {filename}...")

    df = pd.read_csv(filename)

    if df.empty:
        print(f"{mode} dataset is empty!")
        continue

    print(f"Loaded {len(df):,} rows")

    features = [
        "D",
        "H",
        "G",
        "L",
        "C_NaOH0",
        "V_total",
        "N",
        "k_caus",
        "eta_eq"]

    X = df[features]

    y_cost = df["cost"]

    y_eff = df["efficiency"]

    (X_train, X_test,yc_train, yc_test, ye_train, ye_test) = train_test_split( X, y_cost, y_eff, test_size=0.20,random_state=42)


    print("Training cost model...")

    cost_model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)

    cost_model.fit(X_train, yc_train)


    print("Training efficiency model...")
    eff_model = RandomForestRegressor(n_estimators=400,random_state=42,n_jobs=-1)

    eff_model.fit(X_train,ye_train)
    cost_pred = cost_model.predict(X_test)

    eff_pred = eff_model.predict(X_test)

    print("\n==============================")
    print(f"{mode} MODEL PERFORMANCE")
    print("==============================")

    print(f"Cost R²  : {r2_score(yc_test, cost_pred):.4f}")

    print(f"Cost MAE : {mean_absolute_error(yc_test, cost_pred):.2f}")

    print()

    print(f"Efficiency R²  : {r2_score(ye_test, eff_pred):.4f}")

    print(f"Efficiency MAE : {mean_absolute_error(ye_test, eff_pred):.2f}")


    cost_filename = f"cost_model_{mode}.pkl"
    efficiency_filename = f"efficiency_model_{mode}.pkl"
    joblib.dump(cost_model, cost_filename)

    joblib.dump(eff_model, efficiency_filename)

    print("\nModels saved:")

    print(f"  {cost_filename}")

    print(f"  {efficiency_filename}")

    print("\nSetting up Bayesian optimization...")


    if mode == "DAC":

        space = [
            Real(5, 15, name="D"),
            Real(25, 60, name="H"),
            Real(30, 150, name="G"),
            Real(5, 30, name="L"),
            Real(1500, 6000, name="C_NaOH0"),
            Real(100, 500, name="V_total"),
            Integer(3, 6, name="N"),
            Real(0.05, 0.5, name="k_caus"),
            Real(0.90, 0.99, name="eta_eq")
        ]

    else:
        space = [Real(2, 8, name="D"), Real(10, 35, name="H"), Real(0.5, 5, name="G"), Real(0.5, 5, name="L"), Real(500, 3000, name="C_NaOH0"), Real(20, 200, name="V_total"), Integer(2, 5, name="N"), Real(0.05, 0.5, name="k_caus"), Real(0.85, 0.97, name="eta_eq")]

    @use_named_args(space)
    def objective(**params):
        x = pd.DataFrame([params], columns=features)

        predicted_cost = float(cost_model.predict(x)[0])

        predicted_efficiency = float(eff_model.predict(x)[0])

        if not np.isfinite(predicted_cost):
            return 1e9

        if not np.isfinite(predicted_efficiency):
            return 1e9


        penalty = 0.0

        if predicted_efficiency < 90:

            penalty += (
                (90 - predicted_efficiency) ** 2
                * 10000
            )


        if predicted_efficiency > 97:

            penalty += (
                (predicted_efficiency - 98) ** 2
                * 5000
            )

        if predicted_cost > 300:

            penalty += ((predicted_cost - 300) ** 2* 10)


        return predicted_cost + penalty


    print("\nRunning Bayesian Optimization...")

    result = gp_minimize(objective, space, n_calls=60, random_state=42)

    best = dict(zip(features, result.x))


    print("\n==============================")
    print(f"{mode} OPTIMAL DESIGN")
    print("==============================")


    for key, value in best.items():
        print(f"{key:12s}: {value:.3f}")

    print("\nRunning physics verification...")


    (CO2_tpy, actual_cost, actual_efficiency, results) = run_full_model(mode=mode, D=best["D"], H=best["H"], G=best["G"], L=best["L"], C_NaOH0=best["C_NaOH0"], V_total=best["V_total"], N=int(round(best["N"])), k_caus=best["k_caus"], eta_eq=best["eta_eq"])


    print("\n==============================")
    print(f"{mode} VERIFIED SIMULATION")
    print("==============================")
    print(f"CO₂ Captured : {CO2_tpy:,.0f} t/year")

    print(f"Efficiency   : {actual_efficiency:.2f}%")

    print(f"Cost         : ${actual_cost:.2f}/tCO₂")

    print("\n==============================")
    print(f"{mode} FEATURE IMPORTANCE")
    print("==============================")

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": cost_model.feature_importances_
        })


    importance = importance.sort_values("Importance", ascending=False)


    print(importance.to_string(index=False))


    print("\nFinished", mode)


print("\n")
print("=" * 60)
print("ALL MODEL TRAINING COMPLETE")
print("=" * 60)

print("\nExpected model files:")

print("  cost_model_DAC.pkl")
print("  efficiency_model_DAC.pkl")
print("  cost_model_INDUSTRIAL.pkl")
print("  efficiency_model_INDUSTRIAL.pkl")