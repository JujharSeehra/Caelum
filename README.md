# Caelum

## Overview

Caelum is a carbon capture technology project focused on improving the efficiency and economics of **Direct Air Capture (DAC)** through computational modeling, chemical process simulation, and machine learning.

Caelum uses **sodium hydroxide (NaOH)** and **calcium hydroxide (Ca(OH)₂)** in alkaline solutions to capture carbon dioxide through chemical absorption. The captured carbon is subsequently processed through chemical reactors to produce **calcium carbonate (CaCO₃)**.

The project combines first-principles chemical and physical simulation, machine learning prediction, bayesian optimization, process and economic modeling, and interactive Streamlit visualization

The computational models are designed to investigate the performance of Caelum across different operating conditions and identify optimized reactor configurations.

# Project Structure

```text
Caelum/
│
├── Data-Recording/
│   ├── Code-Logger.py
│   ├── carbon_capture_all_runs.csv
│   └── carbon_capture_all_runs.txt
│
├── ML-PROJECT/
│   ├── data_generation.py
│   ├── ML-Optimizer.py
│   └── simulation.py
│
├── README.md
├── requirements.txt
├── install.sh
└── .gitignore
```

> **Note:** Some generated files and deployment-specific files may not be included in the public repository. These files are generated locally during the workflow.

# Requirements

Caelum requires:

* Python 3
* `pip`
* A terminal
* Approximately several GB of available storage may be useful for larger generated datasets

The project uses Python libraries including:

* NumPy
* Pandas
* Scikit-learn
* Scikit-optimize
* Joblib
* Plotly
* Streamlit

The main application and platform for viewing results can be found on Streamlit. 

# Streamlit Dashboard

The Caelum dashboard provides an interactive interface for exploring the carbon-capture system.

The user can select between:

* **DAC**
* **INDUSTRIAL**

and modify process parameters including:

* Absorber diameter
* Absorber height
* Gas flow rate
* Liquid flow rate
* NaOH concentration
* Causticizer volume
* Number of CSTRs
* Reaction rate constant
* Equilibrium conversion

When physics verification is enabled, the dashboard additionally runs the first-principles simulation and displays:

* CO₂ captured per year
* Physics-based capture cost
* Physics-based efficiency
* ML vs. physics comparison
* Cost error
* Annual cost breakdown
* Operating-cost distribution
* Power consumption
* Process visualizations

# Scientific Approach

Caelum models carbon capture using alkaline chemical absorption.

In the capture process, carbon dioxide reacts with sodium hydroxide:

```text
CO₂ + 2NaOH → Na₂CO₃ + H₂O
```

The resulting sodium carbonate can then participate in a causticization process involving calcium hydroxide:

```text
Na₂CO₃ + Ca(OH)₂ → CaCO₃ + 2NaOH
```

This allows the sodium hydroxide to be regenerated while producing calcium carbonate as the resulting carbon-containing product.

The system therefore combines:

**CO₂ absorption → chemical conversion → NaOH regeneration → CaCO₃ production**

The computational model investigates the performance of this process under different physical, chemical, and economic conditions.

# Machine Learning

Caelum uses machine learning as an optimization and prediction layer on top of the underlying process simulation.

Random Forest regression models are trained to approximate the relationships between process parameters and:

1. Capture cost
2. Capture efficiency

Bayesian optimization is subsequently used to search the parameter space for configurations that satisfy performance constraints while minimizing predicted cost.

The resulting configuration is then evaluated using the underlying physics-based model.

This creates a workflow in which machine learning is used for computational optimization while the physical simulation provides verification.

# Important Notes

### Run the scripts in order

Do not launch the Streamlit application before generating the training data and trained models.

The expected workflow is:

```bash
python data_generation.py
python ML-Optimizer.py
streamlit run MLapp.py
```

### Generated files

Some files used by Caelum are generated during execution rather than stored permanently in the repository.

These include machine-learning model files and generated training datasets.

If these files are absent, regenerate them by following Steps 1 and 2 above.

### Virtual environments

Do not commit your local Python virtual environment to GitHub.

The `requirements.txt` file provides the dependencies required to recreate the environment.


# Contact

For questions, collaboration, or inquiries regarding Caelum:

**Jujhar Seehra**
**Email:** [jujharseehra@gmail.com](mailto:jujharseehra@gmail.com)