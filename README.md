# Papyrus-scripts

Collection of scripts to interact with the Papyrus bioactivity dataset.

![alt text](https://raw.githubusercontent.com/OlivierBeq/Papyrus-scripts/refs/heads/master/figures/papyrus_workflow.svg)

<br/>

**Associated Article:** <a href="https://doi.org/10.1186/s13321-022-00672-x">10.1186/s13321-022-00672-x</a>
```
Béquignon OJM, Bongers BJ, Jespers W, IJzerman AP, van de Water B, van Westen GJP.
Papyrus - A large scale curated dataset aimed at bioactivity predictions.
J Cheminform 15, 3 (2023). https://doi.org/10.1186/s13321-022-00672-x
```

**Associated Preprint:** <a href="https://doi.org/10.33774/chemrxiv-2021-1rxhk">10.33774/chemrxiv-2021-1rxhk</a>
```
Béquignon OJM, Bongers BJ, Jespers W, IJzerman AP, van de Water B, van Westen GJP.
Papyrus - A large scale curated dataset aimed at bioactivity predictions.
ChemRxiv. Cambridge: Cambridge Open Engage; 2021;
This content is a preprint and has not been peer-reviewed.
```

## Installation

```bash
pip install papyrus-scripts
``` 

:warning: If pip gives the following error and resolves in import errors
```bash
Defaulting to user installation because normal site-packages is not writeable
```
Then uninstall and reinstalling the library with the following commands:
```bash
pip uninstall -y papyrus-scripts
python -m pip install papyrus-scripts
```

Additional dependencies can be installed to allow:
 - similarity and substructure searches
    ```bash
    conda install FPSim2 openbabel h5py cupy -c conda-forge
    ```

- training DNN models:
    ```bash
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    ```

## Getting started

### The new application programming interface (API)
This new object-oriented API is available since version 2.0.0.

It allows for easier filtering of the Papyrus data and ensures that any data being queried is downloaded.

```python
from papyrus_scripts import PapyrusDataset

data = (PapyrusDataset(version='05.7', plusplus=True) # Downloads the data if needed
        .keep_source(['chembl', 'sharma']) # Keep specific sources
        .keep_quality('high')
        .proteins() # Get the corresponding protein targets
        )
```

### Functional API (legacy)

The functional API requires the data to be downloaded beforehand.<br/>
One can donwload the dataset either with the functional API itself or the command line interface (CLI).

#### Donwloading with the command line interface (CLI)
The following command will download the Papyrus++ bioactivities and protein targets (high-quality Ki and KD data as well as IC50 and EC50 of reproducible assays) for the latest version.
```bash
papyrus download -V latest
```
The following command will donwload the entire set of high-, medium-, and low-quality bioactivities and protein targets along with all precomputed molecular and protein descriptors for version 05.5.
```bash
papyrus download -V 05.5 --more --d all 
```
The following command will download Papyrus++ bioactivities, protein targets and compound structures for both version 05.4 and 05.5.
```bash
papyrus download -V 05.5 -V 05.4 -S 
```

More options can be found using 
```bash
papyrus download --help 
```

By default, the data is downloaded to [pystow](https://github.com/cthoyt/pystow)'s default directory.<br/>
One can override the folder path by specifying the `-o` switch in the above commands.

#### Donwloading with the functional API

```python

from papyrus_scripts import download_papyrus

# Donwload the latest version of the entire dataset with all precomputed descriptors
download_papyrus(version='latest', only_pp=False, structures=True, descriptors='all')
```

#### Querying with the functional API

The query detailed above using the object-oriented API is reproduced below using the functional API.

```python
from papyrus_scripts import (read_papyrus, read_protein_set,
                             keep_quality, keep_source, keep_type,
                             keep_organism, keep_accession, keep_protein_class,
                             keep_match, keep_contains,
                             consume_chunks)

chunk_reader = read_papyrus(version='05.7', plusplus=True, is3d=False, chunksize=1_000_000)
protein_data = read_protein_set(version='05.7')
filter1 = keep_source(data=chunk_reader, source=['chembl', 'sharma'])
filter2 = keep_quality(data=filter1, min_quality='high')
data = consume_chunks(filter2, progress=False)

protein_data = protein_data.set_index('target_id').loc[data.target_id.unique()].reset_index()
```

## Versions of the Papyrus dataset

Different online servers host the Papyrus data based on release and ChEMBL version (table below).

 
| Papyrus version | ChEMBL version |                          Zenodo                           |                            4TU                            |
|:---------------:|:--------------:|:---------------------------------------------------------:|:---------------------------------------------------------:|
|      05.4       |       29       | [:heavy_check_mark:](https://zenodo.org/records/10943992) | [:heavy_check_mark:](https://doi.org/10.4121/16896406.v2) | 
|      05.5       |       30       | [:heavy_check_mark:](https://zenodo.org/records/7019873)  |                            :x:                            |
|      05.6       |       31       | [:heavy_check_mark:](https://zenodo.org/records/7373213)  |                            :x:                            |
|      05.7       |       34       | [:heavy_check_mark:](https://zenodo.org/records/13787633) |                            :x:                            |

Precomputed molecular and protein descriptors along with molecular structures (2D for default set and 3D for low quality set with stereochemistry) are not available for version 05.4 from 4TU but are from Google Drive.

As stated in the pre-print **we strongly encourage** the use of the dataset in which stereochemistry was not considered.
This corresponds to files containing the mention "2D" and/or "without_stereochemistry". 

## Interconversion of the compressed files

The available LZMA-compressed files (*.xz*) may not be supported by some software (e.g. Pipeline Pilot).
<br/>**Decompressing the data is strongly discouraged!**<br/>
Though Gzip files were made available at 4TU for version 05.4, we now provide a CLI option to locally interconvert from LZMA to Gzip and vice-versa.

To convert from LZMA to Gzip (or vice-versa) use the following command:
```bash
papyrus convert -v latest 
```

## Removal of the data

One can remove the Papyrus data using either the CLI or the API.

The following exerts exemplify the removal of all Papyrus data files, including all versions utility files. 
```bash
papyrus clean --remove_root
```

```python
from papyrus_scripts import remove_papyrus

remove_papyrus(papyrus_root=True)
```


## Easy handling of the dataset

Once installed the Papyrus-scripts allow for the easy filtering of the data.<br/>
- Simple examples can be found in the <a href="https://github.com/OlivierBeq/Papyrus-scripts/blob/master/notebook_examples/simple_examples.ipynb">simple_examples.ipynb</a> notebook. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OlivierBeq/Papyrus-scripts/blob/master/notebook_examples/simple_examples.ipynb)
- An example on matching data with the Protein Data Bank can be found in the <a href="https://github.com/OlivierBeq/Papyrus-scripts/blob/master/notebook_examples/matchRCSB.ipynb">simple_examples.ipynb</a> notebook. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OlivierBeq/Papyrus-scripts/blob/master/notebook_examples/matchRCSB.ipynb)
- More advanced examples will be added to the <a href="https://github.com/OlivierBeq/Papyrus-scripts/blob/master/notebook_examples/advanced_querying.ipynb">advanced_querying.ipynb</a> notebook.
## Reproducing results of the pre-print

The scripts used to extract subsets, generate models and obtain visualizations can be found <a href="https://github.com/OlivierBeq/Papyrus-modelling">here</a>.

## Features to come

- [x] Substructure and similarity molecular searches
- [x] ability to use DNN models
- [x] ability to repeat model training over multiple seeds
- [x] y-scrambling
- [ ] adapt models to QSPRpred
 
## Examples to come

- Use of custom grouping schemes for training/test set splitting and cross-validation
- Use custom molecular and protein descriptors (either Python function or file on disk) 


## Logos

Logos can be found under <a href="https://github.com/OlivierBeq/Papyrus-scripts/tree/master/figures/logo">**figures/logo**</a>
Two version exist depending on the background used.

:warning: GitHub does not render the white logo properly in the table below but should not deter you from using it! 

<div class="colored-table">

|                                                          On white background                                                           |                                                             On colored background                                                            |
|:--------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------:|
| <img src="https://raw.githubusercontent.com/OlivierBeq/Papyrus-scripts/refs/heads/master/figures/logo/Papyrus_trnsp-bg.svg" width=200> | <img src="https://raw.githubusercontent.com/OlivierBeq/Papyrus-scripts/refs/heads/master/figures/logo/Papyrus_trnsp-bg-white.svg" width=200> |

</div>
