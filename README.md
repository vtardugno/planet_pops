# NASA *Kepler* Transit Population Simulator

The *Kepler* Transit Population Simulator is a Python-based tool designed to simulate *Kepler* transit observations by generating synthetic exoplanet populations. By specifying population parameters and a stellar catalog, users can generate *Kepler* transit observations.

## Getting Started

Ensure you have the following Python packages installed:
- Numpy
- Matplotlib
- Forecaster (Chen, J., & Kipping, D. 2017, ApJ, 834, 17, doi: 10.3847/1538-4357/834/1/17259)

## Running a Simulation

To simulate a planetary population and compute its observable transits:

```bash
python simulate.py --rcrit {rcrit} --alpha_small {alpha_small} --alpha_big {alpha_big} --sigma {sigma} --sigma_i {sigma_i} --b_m {b_m} --eta_zero {eta_zero} --o r{rcrit}_as{alpha_small}_ab{alpha_big}_s{sigma}_si{sigma_i}_bm{b_m}_ez{eta_zero}"
```

### Outputs

The script outputs two torch tensors per simulation: one contains the input parameters, and the other contains a histogram describing the observed planet multiplicity.
