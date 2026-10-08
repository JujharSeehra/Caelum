# Caelum rerun workflow

This cleaned workflow replaces the old `data_generation.py` + `ML-Optimizer.py` + slider-heavy Streamlit app.

## What changed

- Fixes kg/year vs metric-tonnes/year conversion.
- Uses unit-consistent OH reaction-rate constants in the absorber model.
- Replaces the batch-like causticizer integration with a steady-state CSTR-train conversion and couples regeneration conversion to sustainable capture.
- Uses the 5% Ca(OH)2 excess consistently in the economics.
- Rejects designs outside the model's superficial gas/liquid velocity range.
- Generates DAC and industrial datasets in one run.
- Trains fresh random-forest cost/efficiency surrogate models for both modes.
- Performs differential-evolution optimization directly on the corrected physics/economic model; RF models are retained for diagnostics and feature importance.
- Saves all outputs under `results/` and all trained models under `models/`.
- Streamlit shows only optimal configurations, engineering data, graphs, and downloads.

## Important

Delete/ignore the old files before rerunning:

- `training_data_DAC.csv`
- `training_data_INDUSTRIAL.csv`
- `cost_model_*.pkl`
- `efficiency_model_*.pkl`

The new workflow does not read them.

## Run from a clean terminal

```bash
cd /path/to/Caelum_Rerun_Ready
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
streamlit run MLapp.py
```

`run_pipeline.py` creates:

- `results/training_data_DAC.csv`
- `results/training_data_INDUSTRIAL.csv`
- `results/optimal_config_DAC.json`
- `results/optimal_config_INDUSTRIAL.json`
- `results/optimal_config_DAC.csv`
- `results/optimal_config_INDUSTRIAL.csv`
- `results/optimal_profile_DAC.csv`
- `results/optimal_profile_INDUSTRIAL.csv`
- `results/optimal_summary.csv`
- `results/model_metrics.csv`
- `results/feature_importance_DAC.csv`
- `results/feature_importance_INDUSTRIAL.csv`
- `models/cost_model_DAC.pkl`
- `models/efficiency_model_DAC.pkl`
- `models/cost_model_INDUSTRIAL.pkl`
- `models/efficiency_model_INDUSTRIAL.pkl`

## Fast test before the full run

Temporarily set in `run_pipeline.py`:

```python
NUM_SAMPLES = 250
OPT_MAXITER = 15
```

Run the pipeline once. If it completes, return to:

```python
NUM_SAMPLES = 3000
OPT_MAXITER = 40
```

and run the final calculation.
