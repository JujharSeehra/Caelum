import numpy as np
import pandas as pd

from simulation import run_full_model
MODES = ["DAC", "INDUSTRIAL"]

NUM_SAMPLES = 3000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

ranges = {

    "DAC": {

        "D": (5.0, 15.0),
        "H": (25.0, 60.0),

        "G": (30.0, 150.0),
        "L": (5.0, 30.0),

        "C_NaOH0": (1500.0, 6000.0),
        "V_total": (100.0, 500.0),

        "N": (3, 6),

        "k_caus": (0.05, 0.50),
        "eta_eq": (0.90, 0.99)
    },


    "INDUSTRIAL": {

        "D": (2.0, 8.0),
        "H": (10.0, 35.0),

        "G": (0.5, 5.0),
        "L": (0.5, 5.0),

        "C_NaOH0": (500.0, 3000.0),
        "V_total": (20.0, 200.0),

        "N": (2, 5),

        "k_caus": (0.05, 0.50),
        "eta_eq": (0.85, 0.97)
    }
}

for mode in MODES:

    print("\n" + "=" * 60)
    print(f"GENERATING {mode} DATASET")
    print("=" * 60)

    params = ranges[mode]

    data = []

    successful = 0
    failed = 0

    for i in range(NUM_SAMPLES):

        D = np.random.uniform(*params["D"])
        H = np.random.uniform(*params["H"])
        G = np.random.uniform(*params["G"])
        L = np.random.uniform(*params["L"])

        C_NaOH0 = np.random.uniform(*params["C_NaOH0"])

        V_total = np.random.uniform(*params["V_total"])

        N = np.random.randint(params["N"][0], params["N"][1] + 1)

        k_caus = np.random.uniform(*params["k_caus"])

        eta_eq = np.random.uniform(*params["eta_eq"])


        try:
            (CO2_tpy, cost, efficiency, results) = run_full_model(mode=mode, D=D, H=H, G=G, L=L, C_NaOH0=C_NaOH0, V_total=V_total, N=N, k_caus=k_caus, eta_eq=eta_eq)
            if results == {}:
                failed += 1
                continue

            values = [CO2_tpy, cost, efficiency]

            if not all(np.isfinite(v) for v in values):
                failed += 1
                continue

            if CO2_tpy <= 0:
                failed += 1
                continue

            if cost <= 0:
                failed += 1
                continue

            if efficiency <= 0:
                failed += 1
                continue
            if efficiency > 100:
                failed += 1
                continue
            data.append({
                "D": D,
                "H": H,
                "G": G,
                "L": L,

                "C_NaOH0": C_NaOH0,
                "V_total": V_total,

                "N": N,

                "k_caus": k_caus,
                "eta_eq": eta_eq,

                "CO2_tpy": CO2_tpy,
                "cost": cost,
                "efficiency": efficiency,

                "pressure_drop_Pa":
                    results["pressure_drop_Pa"],

                "pump_power_kW":
                    results["pump_power_kW"],

                "blower_power_kW":
                    results["blower_power_kW"],

                "total_power_kW":
                    results["total_power_kW"],

                "annual_energy_kWh":
                    results["annual_energy_kWh"],

                "installed_cost":
                    results["installed_cost"],

                "annual_cost":
                    results["annual_cost"],

                "lime_cost":
                    results["lime_cost"],

                "electricity_cost":
                    results["electricity_cost"]
            })
            successful += 1


        except Exception as e:

            failed += 1
            if failed <= 10:
                print(f"Failed sample {i + 1}: {e}")


        if (i + 1) % 500 == 0:
            print(
                f"Completed {i + 1:,}/{NUM_SAMPLES:,} | Successful: {successful:,} | Failed: {failed:,}")
    df = pd.DataFrame(data)
    print("\nDataset Summary")
    print(f"Successful rows: {len(df):,}")
    print(f"Failed rows: {failed:,}")
    if df.empty:
        print(f"ERROR: {mode} dataset is empty.")
        continue
    print(df[["D", "H", "G", "L", "CO2_tpy", "cost", "efficiency" ]].describe())

    filename = f"training_data_{mode}.csv"
    df.to_csv(filename,index=False)


    print(f"\nSaved: {filename}")

print("\n" + "=" * 60)
print("DATA GENERATION COMPLETE")
print("=" * 60)