# -*- coding: utf-8 -*-

import os
import sys
from typing import List, Optional, Union

# Optional dependencies
try:
    import gdown
except ImportError as e:
    gdown = e
try:
    import pystow
except ImportError as e:
    pystow = e
from tqdm.auto import tqdm

from .utils.IO import enough_disk_space, assert_sha256sum


def download_papyrus(outdir: Optional[str] = None,
                     nostereo: bool = True,
                     stereo: bool = False,
                     structures: bool = False,
                     descriptors: Union[str, List[str]] = 'all',
                     progress: bool = True):
    """Download the Papyrus data.
    
    :param outdir: directory where Papyrus data is stored (default: pystow's directory)
    :param nostereo: should 2D data be downloaded
    :param stereo: should 3D data be downloaded
    :param structures: should molecule structures be downloaded
    :param descriptors: should molecular and protein descriptors be downloaded
    :param progress: should progress be displayed
    """
    if isinstance(gdown, ImportError) and isinstance(pystow, ImportError):
        print('\nSome required dependencies are missing:\n\tpystow, gdown')
        sys.exit()
    elif isinstance(gdown, ImportError):
        print('\nSome required dependencies are missing:\n\gdown')
        sys.exit()
    elif isinstance(pystow, ImportError):
        print('\nSome required dependencies are missing:\n\tpystow')
        sys.exit()
    files = {'2D_papyrus':
                { 'name': "05.4_combined_set_without_stereochemistry.tsv.xz",
                  'id': "137zOUqUpFFwYGFFpiXEfsAxkGBqUoL8f",
                  'size': 742110788,
                  'sha256': "1a1c946917f77d9a250a181c8ef19bea4d04871915e9e75a615893a2c514684e"},
             '2D_structures':
                { 'name': "05.4_combined_2D_set_without_stereochemistry.sd.xz",
                  'id': "1ihzKOkYrOIrN3D4vUfXrOdDyRpekHfwP",
                  'size': 416640448,
                  'sha256': "4595f726daf12a784049f20e9f9464ed0287af3a22a27f2a919399c535f633fc"},
             '3D_papyrus':
                { 'name': "05.4_combined_set_with_stereochemistry.tsv.xz",
                  'id': "1_givKTu7dKHbIn5xqIhanM2wKZFDVi53",
                  'size': 777395668,
                  'sha256': "56cf389030246d4525bb31cd3dfc9e5ab3afa9613535d1540c71f0f7426c778f"},
             '3D_structures':
                { 'name': "05.4_combined_3D_set_with_stereochemistry.sd.xz",
                  'id': "1rhuztuSS4NllVbFOK_Ob7ifpDSvOV7ED",
                  'size': 446702556,
                  'sha256': "b0f04e066b7ac6b1e1f2a868ff0258b13bd8d3433023ff59c3af58317bfeb3e9"},
             '2D_fingerprint':
                { 'name': "05.4_combined_2D_moldescs_ECFP6.tsv.xz",
                  'id': "1XGyFyxJqfSQu6OlM27hO5RAzh1zezW0Z",
                  'size': 141318356,
                  'sha256': "4ab781cc238107f7c48f1d866eea0e2114068b6512acf74932a5b21958c9ffe0"},
             '3D_fingerprint':
                { 'name': "05.4_combined_3D_moldescs_E3FP.tsv.xz",
                  'id': "1dtyckIi3reC30YSwjMMxRUoYd9DpDZS-",
                  'size': 146751352,
                  'sha256': "2b89027dad8f4e59f007dd082664a7d2a491f4f79d112fb29f14565acedfe4d0"},
             '2D_mordred':
                { 'name': "05.4_combined_2D_moldescs_mordred2D.tsv.xz",
                  'id': "13nr3O9_r7CckTIVk0GcZORb-qWZat_pa",
                  'size': 3085232504,
                  'sha256': "d15bca59f542a6c46528e4f131cb44d8bd6b21440ab139f4175f4327c15c39c6"},
             '3D_mordred':
                { 'name': "05.4_combined_3D_moldescs_mordred3D.tsv.xz",
                  'id': "1pOKh29bZNrfiT-hZ4FOZCjrDXriFRmA9",
                  'size': 2996851908,
                  'sha256': "80fc4f9b2d0b89e68c289c44e9f4df78f4c08e5867cd414d6169a4e1344aead8"},
             '2D_cddd':
                { 'name': "05.4_combined_2D_moldescs_CDDDs.tsv.xz",
                  'id': "1b1KuGN_i2oTD8NeL2uECpZbdhefOaeaa",
                  'size': 3770082588,
                  'sha256': "9bb0d9adba1b812aa05b6391ecbc3f0148f6ed37972a004b13772d08790a9bda"},
             '2D_mold2':
                { 'name': "05.4_combined_2D_moldescs_mold2.tsv.xz",
                  'id': "1U6XVs80hfBcFHYRvcpFDq_Wu9KOujBSt",
                  'size': 1552425452,
                  'sha256': "bdfb0cbb6e9a3d1b62065808fa0e6ce238e04760df62e34ce4f15046810efd82"},
             'proteins':
                { 'name': "05.4_combined_set_protein_targets.tsv.xz",
                  'id': "1hrSb8ouyD9F47ndiuMXlK-VmovrDThnV",
                  'size': 1701316,
                  'sha256': "5f49030509ce188a119910f16054558e1cdd1c70a22d2a1458ec4189f5d1a08e"},
             'proteins_unirep':
                { 'name': "05.4_combined_prot_embeddings_unirep.tsv.xz",
                  'id': "1yYKakgrByeHGZ9PZV5xJbMKbnjM21udE",
                  'size': 138392528,
                  'sha256': "19aa0562c3b695883c5aa8c05ad0934c4b9b851a26550345940d92ed17f36b93"}
            }
    # Define root dir for downloads
    if outdir is not None:
        os.environ['PYSTOW_HOME'] = os.path.abspath(outdir)
    papyrus_root = pystow.module('papyrus')
    # Prepare files to be downloaded
    downloads = set()
    if nostereo:
        downloads.add('2D_papyrus')
        downloads.add('proteins')
        if structures:
            downloads.add('2D_structures')
        if 'mold2' in descriptors or 'all' in descriptors:
            downloads.add('2D_mold2')
        if 'cddd' in descriptors or 'all' in descriptors:
            downloads.add('2D_cddd')
        if 'mordred' in descriptors or 'all' in descriptors:
            downloads.add('2D_mordred')
        if 'fingerprint' in descriptors or 'all' in descriptors:
            downloads.add('2D_fingerprint')
    if stereo:
        downloads.add('3D_papyrus')
        downloads.add('proteins')
        if structures:
            downloads.add('3D_structures')
        if 'mordred' in descriptors or 'all' in descriptors:
            downloads.add('3D_mordred')
        if 'fingerprint' in descriptors or 'all' in descriptors:
            downloads.add('3D_fingerprint')
    if 'unirep' in descriptors or 'all' in descriptors:
        downloads.add('proteins_unirep')
    # Download files
    pbar = tqdm(downloads) if progress else downloads
    for ftype in pbar:
        download = files[ftype]
        dname, did, dsize, dhash = download['name'], download['id'], download['size'], download['sha256']
        if not enough_disk_space(papyrus_root.base.as_posix(), dsize):
            raise IOError(f'not enough disk space for the required {dsize / 2 ** 30:.2f} GiB')
        fpath = papyrus_root.join(name=dname).as_posix()
        # Download file
        correct = False  # ensure file is not corrupted
        retries = 3
        while not correct and retries > 0:
            gdown.cached_download(id=did, path=fpath, quiet=False, resume=True)
            correct = assert_sha256sum(fpath, dhash)
            if not correct:
                os.remove(fpath)
        if retries == 0:
            print('Failed to download {dname} (3 attempts)')