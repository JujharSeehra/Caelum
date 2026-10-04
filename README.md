# Caelum

## Overview

Caelum is a carbon capture technology project focused on improving the efficiency and economics of **Direct Air Capture (DAC)** through computational modeling, chemical process simulation, and machine learning.

Caelum uses **sodium hydroxide (NaOH)** and **calcium hydroxide (Ca(OH)₂)** in alkaline solutions to capture carbon dioxide through chemical absorption. The captured carbon is subsequently processed through chemical reactors to produce **calcium carbonate (CaCO₃)**.

The project combines:

* First-principles chemical and physical simulation
* Machine learning prediction
* Bayesian optimization
* Process and economic modeling
* Interactive Streamlit visualization

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

# Installation

Clone the repository:

```bash
git clone https://github.com/JujharSeehra/Caelum.git
```

Enter the project directory:

```bash
cd Caelum
```

## Option 1 — Automatic Installation

Run:

```bash
chmod +x install.sh
./install.sh
```

This installs the Python dependencies listed in `requirements.txt`.

## Option 2 — Manual Installation

Create a virtual environment:

```bash
python3 -m venv venv
```

Activate it.

### macOS / Linux

```bash
source venv/bin/activate
```

### Windows

```powershell
venv\Scripts\activate
```

Then install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

# Step 1 — Generate Training Data

Navigate into the ML project:

```bash
cd ML-PROJECT
```

Run:

```bash
python data_generation.py
```

This runs the Caelum simulation across a range of operating conditions and generates the datasets used to train the machine-learning models.

The generated datasets include separate data for:

* Direct Air Capture (DAC)
* Industrial carbon capture

These datasets are used by the machine-learning training and optimization process.

Depending on the configuration and number of simulations, this step may take some time.

# Step 2 — Train and Optimize the Machine Learning Models

After training data generation has completed, run:

```bash
python ML-Optimizer.py
```

This performs several stages of computation.

### Bayesian Optimization

After training the models, Caelum uses Bayesian optimization to search for promising process configurations.

The optimization attempts to find configurations that balance:

* High carbon-capture efficiency
* Low cost
* Physical constraints

### Physics Verification

The optimized configuration is then passed back through the first-principles simulation.

This provides an independent physics-based verification of the machine-learning prediction.

Upon completion, the trained model files are generated locally.

These model files are required by the Streamlit application.

# Step 3 — Launch the Streamlit Application

After `ML-Optimizer.py` has completed successfully, launch the Streamlit dashboard.

From the `ML-PROJECT` directory:

```bash
streamlit run MLapp.py
```

Alternatively, from the Caelum root directory:

```bash
streamlit run ML-PROJECT/MLapp.py
```

Streamlit will provide a local URL, typically:

```text
http://localhost:8501
```

Open this address in your web browser.

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

# Reproducing the Complete Workflow

To reproduce Caelum from a fresh installation:

```bash
git clone https://github.com/JujharSeehra/Caelum.git
cd Caelum

chmod +x install.sh
./install.sh

cd ML-PROJECT

python data_generation.py

python ML-Optimizer.py

streamlit run MLapp.py
```

The commands should be executed **in this order**.

The reason for this sequence is that:

```text
data_generation.py
        ↓
creates training datasets

ML-Optimizer.py
        ↓
trains models and performs optimization

MLapp.py
        ↓
loads the trained models and provides the interactive dashboard
```

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