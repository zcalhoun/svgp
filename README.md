# Spatiotemporal SVGP Experiments
This repository contains code and supporting assets for modelling urban heat with sparse variational Gaussian processes (SVGPs). The workflow ingests hourly weather observations, engineers spatiotemporal features, and trains GP models that can be validated or used to generate predictive maps.

## Highlights
- SVGP models for temperature and dewpoint with custom covariance structures.
- Data pipeline that filters, normalises, and weights station observations.
- Training and evaluation loops with weighted likelihood support and GPU acceleration.
- Notebooks and scripts for exploratory analysis, preprocessing, and figure generation.

## Repository Layout
- `main.py` – command-line entry point used for SLURM array jobs.
- `Datasets/` – dataset loader, QC routines, and LGCP-based weighting helpers.
- `src/` – python package exposing models, trainers, and lightweight utilities.
- `scripts/` – auxiliary scripts for preprocessing and experiment management.
- `notebooks/` – exploratory analysis, diagnostics, and figure notebooks.
- `data/`, `experiments/`, `results/`, `figures/` – local working directories for inputs and outputs (not tracked in git).

## Getting Started
1. **Install dependencies**
   ```bash
   python -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```
   > PyTorch and GPyTorch are required; install the CUDA build that matches your system.

2. **Prepare data**  
Probably the most time-consuming component of applying this method is the data preprocessing. You can think about these steps in three-parts:
1. Data collection.
    * Pull Weather Underground data.
    * Pull NLCD data.
    * Pull ERA5 data.
2. Data preparation.
    * Filter weather underground data to extract into CSV format with only the variables we care about.
    * Apply principal component analysis to NLCD data to extract componnet.
    * Linearly interpolate ERA5 data to Weather Underground coordinates.
3. Data collation.
    * Merge WU, NLKD, and ERA5 on coordinates to create complete feature set.

Note -- the only quality control at this step is the Weather Underground QC procedure. The actual training script handles statistical filtering.

3. **Train a model**
This is the main script of interest from the manuscript. This script contains all of the logic needed to actually fit the model. If you just want to use GPyTorch on your data, and it's a relatively small dataset, I recommend checking out [this tutorial](https://docs.gpytorch.ai/en/stable/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html) on the GPyTorch website.

   ```bash
   python main.py \
     --input data/weather_underground \
     --output results/temp_pll \
     --variable tempAvg \
     --likelihood Student \
     --loss PLL \
     --train_size 0.8 \
     --num_epochs 20 \
     --batch_size 512 \
     --num_inducing_points 1000 \
     --ref_data data/era5_reference
   ```
   When `SLURM_ARRAY_TASK_ID` is set, the script derives `(year, month)` automatically. Without SLURM you can set the variable manually (e.g. `SLURM_ARRAY_TASK_ID=24 python main.py …`).

4. **Examine outputs**  
   - `model_YYYY_MM.pt` and `likelihood_YYYY_MM.pt` store learned parameters.  
   - `results_YYYY_MM.json` (for held-out validation) reports MAE/MSE/NLPD and quantile coverage metrics.  
   - `{YYYY}-{MM}.csv` (when training on full data) stores map-ready predictions and uncertainty bounds.

## Development Tips
- Core code lives under `src/`; import everything via `from src import …`.
- `Datasets/` contains code which is specific to how the data is saved on my machine.
- Custom loggers and GP helpers are defined in `src/utils.py`.

## License
Distributed under the MIT License. You may use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, provided the copyright notice and license text are included. The software is supplied “as is” without warranty, and the authors are not liable for damages arising from its use.

## Questions?
If some of this code is not working for you, please do file an issue!