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

Install conda dependencies first:
```bash
conda install rdkit FPSim2 openbabel -c conda-forge
```

Then install PyTorch <a href="https://pytorch.org/get-started/locally/">locally</a>.

Finally install Papyrus-scripts with pip
```bash
pip install https://github.com/OlivierBeq/Papyrus-scripts/tarball/master
``` 
Dependencies that pypi resolves will auto-magically be installed.

## Donwload the dataset

The Papyrus data relating to bioactivities can be found at 4TU: <a href="https://doi.org/10.4121/16896406">10.4121/16896406</a>.
<br/>

To spare disk usage one can download the 4TU data in XZ format on <a href="https://drive.google.com/drive/folders/1Lhw5G6gu_nLzHQoGmnl02uhFsmOgEZ5a?usp=sharing">Google Drive</a> (if using PipelinePilot, stick to the 4TU gz files).<br/>
Precomputed molecular and protein descriptors along with molecular structures (2D for default set and 3D for low quality set with stereochemistry) are also available from <a href="https://drive.google.com/drive/folders/1Lhw5G6gu_nLzHQoGmnl02uhFsmOgEZ5a?usp=sharing">Google Drive</a>.

As stated in the pre-print **we strongly encourage** the use of the dataset in which stereochemistry was not considered.
This corresponds to files containing the mention "2D" and/or "without_stereochemistry". 

## Easy handling of the dataset

Once installed the Papyrus-scripts allow for the easy filtering of the data.<br/>
Simple examples can be found in the <a href="">simple_examples.ipynb</a> notebook.<br/>
More advanced examples be found in the <a href="">advanced_querying.ipynb</a> notebook.

## Reproducing results of the pre-print

The scripts used to extract subsets, generate models and obtain visualizations can be found under the folder 'preprint'.

Please note that, although the subsets and machine-learning models can easily be generated on a laptop, the TMAP visualizations required more than 150 GiB of RAM.

Molecular structures and descriptors as well as  protein descriptors can be accessed on <a href="https://drive.google.com/drive/folders/1Lhw5G6gu_nLzHQoGmnl02uhFsmOgEZ5a?usp=sharing">Google Drive</a>.

## Features to come

- Substructure and similarity molecular searches
- ability to use DNN models
- ability to repeat model training over multiple seeds
- y-scrambling

## Examples to come

- Use of custom grouping schemes for training/test set splitting and cross-validation
- Use custom molecular and protein descriptors (either Python function or file on disk) 
