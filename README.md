<div align="center">
  <img src="https://raw.githubusercontent.com/OlivierBeq/Papyrus-scripts/refs/heads/master/figures/logo/Papyrus_trnsp-bg-white.svg" alt="Papyrus logo" width="200">

  # 📜 Papyrus-scripts

  [![PyPI version](https://img.shields.io/pypi/v/papyrus-scripts.svg)](https://pypi.org/project/papyrus-scripts/)
  [![Supported Python versions](https://img.shields.io/pypi/pyversions/papyrus-scripts.svg)](https://pypi.org/project/papyrus-scripts/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Tests](https://github.com/OlivierBeq/Papyrus-scripts/actions/workflows/ci.yml/badge.svg)](https://github.com/OlivierBeq/Papyrus-scripts/actions/workflows/ci.yml)
  [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
  [![DOI](https://img.shields.io/badge/DOI-10.1186/s13321--022--00672--x-blue)](https://doi.org/10.1186/s13321-022-00672-x)
</div>

A Python library for working with **Papyrus**, a large-scale curated dataset of bioactivity data aimed at machine learning applications. It handles downloading, versioning, and caching the dataset, and provides a fluent API to filter, search, and export exactly the subset you need — without ever loading the full dataset into memory.

## ✨ Features

- 🗂️ **Versioned dataset access** — download and cache specific Papyrus releases from Zenodo or 4TU, with automatic integrity checks.
- 🔗 **Fluent filtering API** — chain quality, source, organism, protein-class and molecular filters over a lazy Polars pipeline; only what you keep gets materialized.
- 🧬 **Proteins, structures & descriptors** — retrieve matching UniProt targets, 2D/3D compound structures, and precomputed molecular/protein descriptors in one call.
- 🔍 **Similarity & substructure search** — build an indexed FPSim2/RDKit search database over the dataset, with CPU, GPU (CUDA), or auto-fallback search engines.
- 🧠 **DNN-ready** — train PyTorch-based QSAR/PCM models on curated subsets, with y-scrambling and repeated seeds built in.
- 💻 **CLI included** — download, convert and clean up dataset files without writing any Python.
- 🔄 **Format-friendly** — transparent LZMA ↔ Gzip conversion and Parquet caching for tools that don't handle `.xz`.

## 📦 Installation

```bash
pip install papyrus-scripts
```

<details>
<summary><strong>⚠️ Troubleshooting <code>pip</code> installation</strong></summary>

If you see `Defaulting to user installation because normal site-packages is not writeable` followed by import errors, reinstall with:
```bash
pip uninstall -y papyrus-scripts
python -m pip install papyrus-scripts
```
</details>

Optional extras enable additional functionality:

| Extra | Enables |
|---|---|
| `papyrus-scripts[subsim]` | CPU similarity & substructure search (`tables`, `FPSim2`) |
| `papyrus-scripts[gpu]` | GPU-accelerated similarity search (`cupy`) |
| `papyrus-scripts[dnn]` | DNN model training (`torch`, `skorch`) |
| `papyrus-scripts[all]` | Everything above |

> **Note:** `openbabel` (needed only for FP2/FP3/FP4 fingerprints) must be installed via conda-forge, not pip, when used alongside RDKit/FPSim2/cupy in the same environment: `conda install -c conda-forge openbabel`.

## 🛠️ Requirements

- Python 3.11+
- [RDKit](https://www.rdkit.org/docs/Install.html)

## 💡 Usage

### Quickstart: the object-oriented API

The recommended way to interact with the dataset. It downloads and caches data automatically as needed.

```python
from papyrus_scripts import PapyrusDataset

dataset = PapyrusDataset(version='2024.09.2', plusplus=True)

filtered = (dataset
            .keep_source(['chembl', 'sharma'])
            .keep_quality('high'))

df = filtered.to_dataframe()
proteins = filtered.proteins().to_dataframe()
```

### Downloading data: the CLI

```bash
# Download Papyrus++ bioactivities & targets for the latest version
papyrus download -V latest

# Download the full (all-quality) dataset with all precomputed descriptors for a specific revision
papyrus download -V 2022.11.3 --more -d all

# Download Papyrus++ data & compound structures for two versions
papyrus download -V 2022.11.3 -V 2022.04.2 -S

papyrus download --help
```

By default, data is downloaded to [pystow](https://github.com/cthoyt/pystow)'s home directory; override it with `-o`.

<details>
<summary><strong>Legacy functional API</strong></summary>

```python
from papyrus_scripts import (read_papyrus, read_protein_set,
                              keep_quality, keep_source, consume_chunks)

chunks = read_papyrus(version='2024.09.2', plusplus=True, chunksize=1_000_000)
proteins = read_protein_set(version='2024.09.2')

filtered = keep_quality(keep_source(chunks, source=['chembl', 'sharma']), min_quality='high')
df = consume_chunks(filtered)
```
</details>

<details>
<summary><strong>Similarity & substructure search</strong></summary>

```python
from papyrus_scripts.subsim_search import FPSubSim2

fpss = FPSubSim2()
fpss.create_from_papyrus(version='2024.09.2', njobs=-1)  # builds a search database using all CPU cores

# cuda=False (default, CPU) | True (GPU, raises if unavailable) | 'auto' (GPU with CPU fallback)
engine = fpss.get_similarity_lib(cuda='auto')
hits = engine.similarity('CCO', threshold=0.7)

sub_lib = fpss.get_substructure_lib()
matches = sub_lib.substructure('c1ccccc1')
```
</details>

## 📊 Dataset versions

Papyrus releases are hosted on different servers depending on release and ChEMBL version:

| Papyrus version | Revisions | Legacy alias | ChEMBL version | Zenodo | 4TU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **2022.04** | 2 | 05.4 | 29 | ✓ | ✓ |
| **2022.08** | 3 | 05.5 | 30 | ✓ | ✗ |
| **2022.11** | 4 | 05.6 | 31 | ✓ | ✗ |
| **2024.09** | 2 | 05.7 | 34 | ✓ | ✗ |

> **Note:** precomputed descriptors and 2D/3D structures are not available for 2022.04 (05.4) on 4TU, but are on Zenodo. For machine learning use cases, we recommend the datasets without stereochemistry (files marked "2D" and/or "without_stereochemistry").

## ⚙️ Advanced utilities

<details>
<summary><strong>Compression conversion</strong></summary>

Data is distributed as LZMA-compressed files (`.xz`), which some tools (e.g. Pipeline Pilot) don't support. Convert to Gzip (or back) without manually decompressing:

```bash
papyrus convert -v latest
```
</details>

<details>
<summary><strong>Removing downloaded data</strong></summary>

```bash
papyrus clean --remove_root
```
```python
from papyrus_scripts import remove_papyrus

remove_papyrus(papyrus_root=True)
```
</details>

## 📚 Learn more

- [`simple_examples.ipynb`](https://github.com/OlivierBeq/Papyrus-scripts/blob/master/notebook_examples/simple_examples.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OlivierBeq/Papyrus-scripts/blob/master/notebook_examples/simple_examples.ipynb)
- [`matchRCSB.ipynb`](https://github.com/OlivierBeq/Papyrus-scripts/blob/master/notebook_examples/matchRCSB.ipynb) — matching Papyrus data against the Protein Data Bank [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OlivierBeq/Papyrus-scripts/blob/master/notebook_examples/matchRCSB.ipynb)
- [`advanced_querying.ipynb`](https://github.com/OlivierBeq/Papyrus-scripts/blob/master/notebook_examples/advanced_querying.ipynb)
- To reproduce the models, extraction pipeline and visualizations from the original publication, see [Papyrus-modelling](https://github.com/OlivierBeq/Papyrus-modelling).

## 🖋️ Citation

If you use `papyrus-scripts` or the Papyrus dataset in your research, please cite:

```bibtex
@article{Bequignon2023Papyrus,
  title={Papyrus - A large scale curated dataset aimed at bioactivity predictions},
  author={B{\'e}quignon, Olivier J.M. and Bongers, Bart J. and Jespers, Willem and IJzerman, Adriaan P. and van de Water, Bob and van Westen, Gerard J.P.},
  journal={Journal of Cheminformatics},
  volume={15},
  number={3},
  year={2023},
  doi={10.1186/s13321-022-00672-x}
}
```

## 📄 License

This project is licensed under the [MIT License](LICENSE).
