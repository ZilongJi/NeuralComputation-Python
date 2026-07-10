# Head Direction Ring Attractor Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ZilongJi/NeuralComputation-Python/blob/main/Module1_HeadDirectionCircuit/head_direction_ring_attractor_tutorial.ipynb)

This repository contains a step-by-step Jupyter notebook for a hands-on tutorial on head direction cells and ring attractor models.

The main teaching file is [head_direction_ring_attractor_tutorial.ipynb](head_direction_ring_attractor_tutorial.ipynb).

## What The Tutorial Covers

The notebook is organized into three parts:

1. Empirical data visualization
   Students inspect real head direction data, raster plots, tuning curves, and low-dimensional population structure.
2. A single-layer ring attractor
   Students build a symmetric recurrent network and test whether it can hold a stable activity bump.
3. A velocity integrator
   Students add asymmetric angular-velocity inputs, move the bump around the ring, and explore mini-projects such as angular velocity tuning and velocity-gain calibration.

## Repository Contents

- [head_direction_ring_attractor_tutorial.ipynb](head_direction_ring_attractor_tutorial.ipynb): the main workshop notebook
- [utils.py](utils.py): helper functions for plotting and analysis
- [NeuralData/EmpiricalHDData.npz](NeuralData/EmpiricalHDData.npz): empirical head direction dataset used in Section A
- [NeuralData/HDC_binned_Dataset_1_A3701-191119.npz](NeuralData/HDC_binned_Dataset_1_A3701-191119.npz): binned dataset used for population analyses
- [Figures/recordingsite.png](Figures/recordingsite.png): recording-site figure used in the notebook
- [Figures/SkaggsModel.png](Figures/SkaggsModel.png): schematic used in the velocity-integrator section

## Running Locally

Install the core Python packages:

```bash
python3 -m pip install numpy matplotlib scipy scikit-learn seaborn
```

Then launch Jupyter and open the notebook:

```bash
jupyter notebook
```

## Running In Colab

This tutorial is designed to work well in Google Colab once the repository is uploaded to GitHub.

Before using Colab:

- keep all file paths relative, for example `NeuralData/EmpiricalHDData.npz`
- make sure `utils.py`, `NeuralData/`, and `Figures/` are included in the repository
- add an installation cell near the top of the notebook if needed:

```python
!pip install -q numpy matplotlib scipy scikit-learn seaborn
```

## Teaching Notes

The notebook already includes:

- short conceptual prompts
- two discussion questions
- mini-project ideas for angular velocity tuning and velocity-gain calibration

The overall aim is to help students develop intuition for how recurrent circuits can maintain and update a representation of head direction.
