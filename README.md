# EXBYCost
a fast, flexible techno-economic modeling tool for estimating extraction costs from food processing byproducts.
Built with Streamlit and powered by the [BioSTEAM](https://github.com/BioSTEAMDevelopmentGroup/biosteam) modelling framework.

## Features

- **Single Run mode** — configure process conditions and get a full TEA report including capital costs, operating costs, and minimum product selling price
- **Parameter Sweep mode** — vary one or more inputs over a range (linear or log spacing), run all combinations automatically, and download results as a CSV with live progress and ETA
- Multiple feedstock types selectable from a dropdown
- Choice of reactor type: conventional, ultrasound-assisted, or microwave-assisted
- Customisable process conditions: feed flow rate, solvent, temperature, extraction time, and more
- Adjustable economic parameters: IRR, depreciation method, plant start year, operating costs
- Detailed capital and operating cost breakdowns

## How to Use

1. Open the app in your browser
2. Select a feedstock from the dropdown in the sidebar
3. Configure process settings (solvent, temperature, reactor type, flow rates, etc.)
4. Adjust TEA settings (start year, IRR, depreciation method, etc.)
5. Choose a mode:
   - **Single Run** — click *Run Simulation* to generate the full TEA report
   - **Parameter Sweep** — tick the parameters to vary, set their ranges and number of points, then click *Run Sweep* and download the CSV when complete

## Local Installation

### Requirements
- Python 3.10 or higher
- The local module files listed below (must be in the same directory as `extraction_tea_tool.py`)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run extraction_tea_tool.py
```

## Repository Structure

```
├── extraction_tea_tool.py       # Main Streamlit app
├── SolidSolventExtractor.py     # Custom solid-solvent extractor unit
├── Mill.py                      # Custom mill unit
├── biosteam_proxy_finder.py     # BioSTEAM proxy assignment utilities
├── pricing.py                   # Solvent pricing and utility cost functions
├── temperature_thresholds.py    # Chemical temperature threshold classification
└── requirements.txt             # Python dependencies
```

## Dependencies

Key packages used:

- [Streamlit](https://streamlit.io/) — web app framework
- [BioSTEAM](https://github.com/BioSTEAMDevelopmentGroup/biosteam) — biorefinery simulation and TEA
- [ThermoSTEAM](https://github.com/BioSTEAMDevelopmentGroup/thermosteam) — thermodynamic engine
- [flexsolve](https://github.com/yoelcortes/flexsolve) — numerical solvers
- NumPy, SciPy, pandas, Matplotlib, Pillow

## Built With

[BioSTEAM](https://github.com/BioSTEAMDevelopmentGroup/biosteam) — Biological Systems Modeling Framework
