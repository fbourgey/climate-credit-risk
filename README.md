# Assessing Climate Risks of a Large Credit Portfolio

[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/bloomberg/climate-credit-risk/badge)](https://scorecard.dev/viewer/?uri=github.com/bloomberg/climate-credit-risk)
[![Lint](https://github.com/bloomberg/climate-credit-risk/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/marketplace/actions/super-linter)
[![Contributor-Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-fbab2c.svg)](CODE_OF_CONDUCT.md)

This repository contains the code to reproduce the results of the paper
[An Efficient SSP-based Methodology for Assessing Climate
Risks of a Large Credit Portfolio](https://hal.science/hal-04665712/document).

## Installation

To set up the environment and install dependencies, follow these steps:

- Clone the repository and navigate to the project directory:

  ```bash
  git clone https://github.com/bloomberg/climate-credit-risk.git
  cd climate-credit-risk
  ```

- Create a virtual environment:

  ```bash
  python3 -m venv .venv
  ```

- Activate the virtual environment:

  ```bash
  source .venv/bin/activate
  ```

- Install the required dependencies:

  ```bash
  pip install .
  ```

After setting up the environment, you can run the scripts and notebooks in this
repository to reproduce the results presented in the paper.

## Repository Structure

The repository is organized as follows:

- **`firm.py`**: Contains the `Firm` class, which models a single firm's
  optimal carbon emission strategy.
- **`utils.py`**: Utility functions and constants used across the project.
- **`firm.ipynb`**: Notebook for single firm analysis.
- **`opt_emission_decomp.ipynb`**: Notebook for optimal emission decomposition
  for a specific firm, scenario, and sector.
- **`pca.ipynb`**: Notebook to investigate the PCA approximation.
- **`portfolio.ipynb`**: Notebook to analyze and visualize the climate risks
  of a credit portfolio.
- **`rhs_l1_error.ipynb`**: Notebook to study the L1 error between the PCA
  loss and the exact loss.

## License

Distributed under the `Apache-2.0` license. See [LICENSE](LICENSE) for more
information.
