# SVGP
This repository contains code to model urban heat with sparse variational Gaussian processes.

## Code organization

The repository follows a `src/` layout and exposes a `spat_temp` package:

- `main.py` — CLI entrypoint for training or map generation jobs.
- `src/spat_temp/data/` — dataset loaders, QC utilities, and raster dataloaders.
- `src/spat_temp/models/` — SVGP model definitions and likelihood factories.
- `src/spat_temp/training/` — training loops and evaluation helpers.
- `src/spat_temp/utils/` — shared helpers (logging, GP loss helpers, etc.).

Legacy modules (`Datasets/`, `Models/`, `Trainers/`) remain as thin shims that forward to the new package to keep older scripts working.

## License

This project is distributed under the MIT License. You are free to use, modify, and redistribute the code—even for commercial purposes—as long as you include the original copyright notice and license text. The software is provided “as is” without warranty; the authors are not liable for any damages resulting from its use.
