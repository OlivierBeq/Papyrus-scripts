# Papyrus-scripts

Collection of scripts to interact with the Papyrus bioactivity dataset.

![alt text](figures/papyrus_workflow.png?raw=true)

<br/>

**Associated Preprint:** <a href="https://doi.org/10.33774/chemrxiv-2021-1rxhk">10.33774/chemrxiv-2021-1rxhk</a>
```
Béquignon OJM, Bongers BJ, Jespers W, IJzerman AP, van de Water B, van Westen GJP.
Papyrus - A large scale curated dataset aimed at bioactivity predictions.
ChemRxiv. Cambridge: Cambridge Open Engage; 2021;
This content is a preprint and has not been peer-reviewed.
```


## Installation

The Papyrus scripts require dependencies, a few of which can only be installed via conda to the best of our knowledge. 

1. Install conda dependencies first:
```bash
conda install rdkit FPSim2 openbabel "h5py<3.2" cupy pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c conda-forge
```

2. Then install Papyrus-scripts with pip
```bash
pip install https://github.com/OlivierBeq/Papyrus-scripts/tarball/master
``` 
Dependencies that PyPI resolves will auto-magically be installed.

:warning: If pip gives the following error and resolves in import errors
```bash
Defaulting to user installation because normal site-packages is not writeable
```
Then uninstall and reinstalling the library with the following commands:
```bash
pip uninstall -y papyrus-scripts
python -m pip install https://github.com/OlivierBeq/Papyrus-scripts/tarball/master
```

## Donwload the dataset

The Papyrus data can be found at different locations based on release and ChEMBL version (table below).
**The use of the command line interface is strongly recommended to download the data.**
 
| Papyrus version | ChEMBL version | 4TU | Google Drive |
| :--: | :--: | :--: | :--: |
| 05.4 | 29 |  [:heavy_check_mark:](https://doi.org/10.4121/16896406.v2) | [:heavy_check_mark:](https://drive.google.com/drive/folders/1Lhw5G6gu_nLzHQoGmnl02uhFsmOgEZ5a?usp=sharing) | 
| 05.5 | 30 | :x: | [:heavy_check_mark:](https://drive.google.com/drive/folders/1BrCx0lN1YVvjgXOOaJZHJ7DBrLqFAbWV?usp=sharing) |

For Pipeline Pilot users, the 4TU data (gzip format) is advised.
Otherwise, the Google drive  data (xz format) is recommended.

Precomputed molecular and protein descriptors along with molecular structures (2D for default set and 3D for low quality set with stereochemistry) are also available from Google Drive.

As stated in the pre-print **we strongly encourage** the use of the dataset in which stereochemistry was not considered.
This corresponds to files containing the mention "2D" and/or "without_stereochemistry". 

## Easy handling of the dataset

Once installed the Papyrus-scripts allow for the easy filtering of the data.<br/>
- Simple examples can be found in the <a href="notebook_examples/simple_examples.ipynb">simple_examples.ipynb</a> notebook. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OlivierBeq/Papyrus-scripts/blob/master/notebook_examples/simple_examples.ipynb)
- An example on matching data with the Protein Data Bank can be found in the <a href="notebook_examples/matchRCSB.ipynb">simple_examples.ipynb</a> notebook. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OlivierBeq/Papyrus-scripts/blob/master/notebook_examples/matchRCSB.ipynb)
- More advanced examples will be added to the <a href="notebook_examples/advanced_querying.ipynb">advanced_querying.ipynb</a> notebook.
## Reproducing results of the pre-print

The scripts used to extract subsets, generate models and obtain visualizations can be found <a href="https://github.com/OlivierBeq/Papyrus-modelling">here</a>.

## Features to come

- ~~Substructure and similarity molecular searches~~
- ~~ability to use DNN models~~
- ~~ability to repeat model training over multiple seeds~~
- y-scrambling
 
## Examples to come

- Use of custom grouping schemes for training/test set splitting and cross-validation
- Use custom molecular and protein descriptors (either Python function or file on disk) 
