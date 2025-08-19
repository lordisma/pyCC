<h2 align="center">Clustlib: Constrained Clustering Library</h2>

<p align="center">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://www.python.org"><img alt="Python version: 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white"></a>
<a href="https://codecov.io/github/lordisma/pyCC" ><img src="https://codecov.io/github/lordisma/pyCC/graph/badge.svg?token=66LN2RBW0S"/></a>
<a href="https://python-poetry.org/"><img src="https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json" /></a>
</p>

> [!NOTE]
> 🚧 Currently the project is a **Work in progress**, but you can help us to fix it.
>
> Please refer to our [contributing documenation](docs/CONTRIBUTING.md) to know how can you start helping us

> [!WARNING]
> This Library is a Work in progress at the moment, most use cases may not be contempleted or supported yet. If you find something feel free to create a bug report.
>
> In the same way if you think we are missing something important create a feature request

## What it is?

**Clustlib** is a Python library that provides a unified framework for **constrained clustering algorithms**.

Unlike traditional clustering, where groups are formed purely from data similarity, constrained clustering integrates **prior knowledge** in the form of *must-link* and *cannot-link* constraints.

This library includes implementations of both **classic algorithms** and **state-of-the-art metaheuristics**, enabling researchers and practitioners to experiment, compare, and extend methods in a common environment.

If you want to know more deep in our [documentation](https://lordisma.github.io/pyCC/)

## ✨ Key Features

- Unified Python API across all algorithms
- Support for **must-link** and **cannot-link** constraints
- Scikit-learn inspired design for easy integration
- Implementations from different families:
  - **K-Means based methods**
  - **Bayesian models**
  - **Evolutionary and metaheuristic approaches**
- Modular and extensible codebase

## 🚀 Installation

```bash
pip install clustlib
```

## Basic Usage

```python
from clustlib.kmeans import COPKMeans

constraints = np.zeros((X.shape[0], X.shape[0]))

# Example: COP-KMeans
model = COPKMeans(n_clusters=3, constraints=constraints)
model.fit(X)
labels = model.labels_
```
