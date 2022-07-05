# -*- coding: utf-8 -*-

import os
from typing import List, Optional, Union

import requests
import pystow
from tqdm.auto import tqdm

from .utils.IO import get_disk_space, enough_disk_space, assert_sha256sum, read_jsonfile, write_jsonfile


def download_papyrus(outdir: Optional[str] = None,
                     version: Union[str, List[str]] = 'latest',
                     nostereo: bool = True,
                     stereo: bool = False,
                     structures: bool = False,
                     descriptors: Union[str, List[str]] = 'all',
                     progress: bool = True,
                     disk_margin: float = 0.10) -> None:
    """Download the Papyrus data.
    
    :param outdir: directory where Papyrus data is stored (default: pystow's directory)
    :param version: version of the dataset to be downloaded
    :param nostereo: should 2D data be downloaded
    :param stereo: should 3D data be downloaded
    :param structures: should molecule structures be downloaded
    :param descriptors: should molecular and protein descriptors be downloaded
    :param progress: should progress be displayed
    :param disk_margin: percent of free disk space to keep
    """
    CHUNKSIZE = 10 * 1048576  # 10 MB
    RETRIES = 3
    files = {'05.4': {'license':
                          {'name': 'LICENSE.txt',
                           'url': 'https://drive.google.com/uc?id=1J-7FJgb_ufPYQsv9uVUmwPZB1vvjUCtO&confirm=t',
                           'size': 20138,
                           'sha256': '3b2890eacd851373001c4a14623458e3adaf1b1967939aa9c38a318e28d61c00'},
                      'readme':
                          {'name': 'README.txt',
                           'url': 'https://drive.google.com/uc?id=1J5NaEs8TLQ9mn9OnUQuX1dVpzeiPZDvo&confirm=t',
                           'size': 8743,
                           'sha256': 'f552ae0b58121b20c9aefcce0737e5f31240d72676dc9ec559f97585aceb33ad'},
                      'data_types':
                          {'name': "data_types.json",
                           'url': "https://drive.google.com/uc?id=1Fayrxjezw3WKIqUX3Jyjz_1zavmCP-ae&confirm=t",
                           'size': 450559,
                           'sha256': "d80a5810d99b62680ee1a214df5d5a30f505ec335a0c221194efb91d1c23913e"},
                      'data_size':
                          {'name': "data_size.json",
                           'url': "https://drive.google.com/uc?id=1IyQGMuwZ8D7cD7NCKr3s9Rh4vJf9dWC-&confirm=t",
                           'size': 299,
                           'sha256': "f52004854a6322df634594bd42823b0b1fdae9c1082f85164c071f46bdac8019"},
                      '2D_papyrus':
                          {'name': "05.4_combined_set_without_stereochemistry.tsv.xz",
                           'url': "https://drive.google.com/uc?id=137zOUqUpFFwYGFFpiXEfsAxkGBqUoL8f&confirm=t",
                           'size': 742110788,
                           'sha256': "1a1c946917f77d9a250a181c8ef19bea4d04871915e9e75a615893a2c514684e"},
                      '2D_structures':
                          {'name': "05.4_combined_2D_set_without_stereochemistry.sd.xz",
                           'url': "https://drive.google.com/uc?id=1ihzKOkYrOIrN3D4vUfXrOdDyRpekHfwP&confirm=t",
                           'size': 416640448,
                           'sha256': "4595f726daf12a784049f20e9f9464ed0287af3a22a27f2a919399c535f633fc"},
                      '3D_papyrus':
                          {'name': "05.4_combined_set_with_stereochemistry.tsv.xz",
                           'url': "https://drive.google.com/uc?id=1_givKTu7dKHbIn5xqIhanM2wKZFDVi53&confirm=t",
                           'size': 777395668,
                           'sha256': "56cf389030246d4525bb31cd3dfc9e5ab3afa9613535d1540c71f0f7426c778f"},
                      '3D_structures':
                          {'name': "05.4_combined_3D_set_with_stereochemistry.sd.xz",
                           'url': "https://drive.google.com/uc?id=1rhuztuSS4NllVbFOK_Ob7ifpDSvOV7ED&confirm=t",
                           'size': 446702556,
                           'sha256': "b0f04e066b7ac6b1e1f2a868ff0258b13bd8d3433023ff59c3af58317bfeb3e9"},
                      '2D_fingerprint':
                          {'name': "05.4_combined_2D_moldescs_ECFP6.tsv.xz",
                           'url': "https://drive.google.com/uc?id=1XGyFyxJqfSQu6OlM27hO5RAzh1zezW0Z&confirm=t",
                           'size': 141318356,
                           'sha256': "4ab781cc238107f7c48f1d866eea0e2114068b6512acf74932a5b21958c9ffe0"},
                      '3D_fingerprint':
                          {'name': "05.4_combined_3D_moldescs_E3FP.tsv.xz",
                           'url': "https://drive.google.com/uc?id=1dtyckIi3reC30YSwjMMxRUoYd9DpDZS-&confirm=t",
                           'size': 146751352,
                           'sha256': "2b89027dad8f4e59f007dd082664a7d2a491f4f79d112fb29f14565acedfe4d0"},
                      '2D_mordred':
                          {'name': "05.4_combined_2D_moldescs_mordred2D.tsv.xz",
                           'url': "https://drive.google.com/uc?id=13nr3O9_r7CckTIVk0GcZORb-qWZat_pa&confirm=t",
                           'size': 3085232504,
                           'sha256': "d15bca59f542a6c46528e4f131cb44d8bd6b21440ab139f4175f4327c15c39c6"},
                      '3D_mordred':
                          {'name': "05.4_combined_3D_moldescs_mordred3D.tsv.xz",
                           'url': "https://drive.google.com/uc?id=1pOKh29bZNrfiT-hZ4FOZCjrDXriFRmA9&confirm=t",
                           'size': 2996851908,
                           'sha256': "80fc4f9b2d0b89e68c289c44e9f4df78f4c08e5867cd414d6169a4e1344aead8"},
                      '2D_cddd':
                          {'name': "05.4_combined_2D_moldescs_CDDDs.tsv.xz",
                           'url': "https://drive.google.com/uc?id=1b1KuGN_i2oTD8NeL2uECpZbdhefOaeaa&confirm=t",
                           'size': 3770082588,
                           'sha256': "9bb0d9adba1b812aa05b6391ecbc3f0148f6ed37972a004b13772d08790a9bda"},
                      '2D_mold2':
                          {'name': "05.4_combined_2D_moldescs_mold2.tsv.xz",
                           'url': "https://drive.google.com/uc?id=1U6XVs80hfBcFHYRvcpFDq_Wu9KOujBSt&confirm=t",
                           'size': 1552425452,
                           'sha256': "bdfb0cbb6e9a3d1b62065808fa0e6ce238e04760df62e34ce4f15046810efd82"},
                      'proteins':
                          {'name': "05.4_combined_set_protein_targets.tsv.xz",
                           'url': "https://drive.google.com/uc?id=1hrSb8ouyD9F47ndiuMXlK-VmovrDThnV&confirm=t",
                           'size': 1701316,
                           'sha256': "5f49030509ce188a119910f16054558e1cdd1c70a22d2a1458ec4189f5d1a08e"},
                      'proteins_unirep':
                          {'name': "05.4_combined_prot_embeddings_unirep.tsv.xz",
                           'url': "https://drive.google.com/uc?id=1yYKakgrByeHGZ9PZV5xJbMKbnjM21udE&confirm=t",
                           'size': 138392528,
                           'sha256': "19aa0562c3b695883c5aa8c05ad0934c4b9b851a26550345940d92ed17f36b93"}
                      },
             '05.5': {'license':
                          {'name': 'LICENSE.txt',
                           'url': 'https://drive.google.com/uc?id=1BxgAHXyskKoqeoVNigiqH2GTVJ5-B0GR&confirm=t',
                           'size': 20138,
                           'sha256': '3b2890eacd851373001c4a14623458e3adaf1b1967939aa9c38a318e28d61c00'},
                      'readme':
                          {'name': 'README.txt',
                           'url': 'https://drive.google.com/uc?id=12ydPZ3G9Nd4UX9U_yS230sknUwQkptd3&confirm=t',
                           'size': 8962,
                           'sha256': 'def789eac96b38ad4074f1b2defbe204f74b7b45349ff6ed37034a65f8527782'},
                      'data_types':
                          {'name': "data_types.json",
                           'url': "https://drive.google.com/uc?id=19EIDDypF9GvCvXVcOjQ2k5ZDQ3wy4B0B&confirm=t",
                           'size': 450678,
                           'sha256': "d38f0b6b53f0450c5530b5bf44d8a7d0bb85417f22b7c818237e3346fe68149c"},
                      'data_size':
                          {'name': "data_size.json",
                           'url': "https://drive.google.com/uc?id=1z504RsrLxN3RJN4EfGXsHe50uAoEr2lK&confirm=t",
                           'size': 299,
                           'sha256': "058783ab4416771618002c6cc7bd4621fb75db86b541244793a565031f73942d"},
                      '2D_papyrus':
                          {'name': "05.5_combined_set_without_stereochemistry.tsv.xz",
                           'url': "https://drive.google.com/uc?id=15zbhi5S8e-xFe-V3KJhqPPaMCz00dVrV&confirm=t",
                           'size': 718601992,
                           'sha256': "04ecaea97c09d02dbde809ad99ea2127fc3997a4e3b200b56dee85c30801890a"},
                      '2D_structures':
                          {'name': "05.5_combined_2D_set_without_stereochemistry.sd.xz",
                           'url': "https://drive.google.com/uc?id=1FiIe5vytjf5aUkmuvwRAMBHxvZIbhgML&confirm=t",
                           'size': 399767580,
                           'sha256': "2e088ca662c5c33c5fc018c42c9c21e918ec167f1129a0a11fbf9c72888e8be6"},
                      '3D_papyrus':
                          {'name': "05.5_combined_set_with_stereochemistry.tsv.xz",
                           'url': "https://drive.google.com/uc?id=1d4zjI8FGIYi6hQkigNKs9bxRGrQgwLp9&confirm=t",
                           'size': 690498416,
                           'sha256': "822aca70ccf4c19879ae45dfa16de5fc29c3ee08b25739e7a087899652af7dd9"},
                      '3D_structures':
                          {'name': "05.5_combined_3D_set_with_stereochemistry.sd.xz",
                           'url': "https://drive.google.com/uc?id=1s4ShxWrTGRFa8DRkrpKw-4N-x-JRnlrI&confirm=t",
                           'size': 492426264,
                           'sha256': "a4a5355ffc56de8d914c2ad281d10c227171c27e4d6c250daad14a16280cf136"},
                      '2D_fingerprint':
                          {'name': "05.5_combined_2D_moldescs_ECFP6.tsv.xz",
                           'url': "https://drive.google.com/uc?id=10DpfcfVXaOVwPUOs-Qde2rUfS4M1kOfi&confirm=t",
                           'size': 97818228,
                           'sha256': "3d626b4295cfbe73877157d8eea84b911a3cb60bf9571165d88c00cc0b0880d2"},
                      '3D_fingerprint':
                          {'name': "05.5_combined_3D_moldescs_E3FP.tsv.xz",
                           'url': "https://drive.google.com/uc?id=1E6nw2WhVJGwRIOVif005LwWlt9shxGLn&confirm=t",
                           'size': 114052016,
                           'sha256': "446fe36d50487f29a2d7402a53cc661097e884dc0df8ffd278646dba6708cb65"},
                      '2D_mordred':
                          {'name': "05.5_combined_2D_moldescs_mordred2D.tsv.xz",
                           'url': "https://drive.google.com/uc?id=1D-Uc7tKrfuDY_Cq_NSVVqxC7mrVGQvzO&confirm=t",
                           'size': 2936434876,
                           'sha256': "bcef94b1c04a1e7d8f9da11ad87e598e19932548a8ea4f00029c2f3a89672ff4"},
                      '3D_mordred':
                          {'name': "05.5_combined_3D_moldescs_mordred3D.tsv.xz",
                           'url': "https://drive.google.com/uc?id=1v-ibHAlQlhJrmEXwbmQ1rsClcyC_vV2s&confirm=t",
                           'size': 3206020732,
                           'sha256': "e6ffd0858f85217b57c4a88619e5f41d7f6bae16a9948612872162e54d3231dc"},
                      '2D_cddd':
                          {'name': "05.5_combined_2D_moldescs_CDDDs.tsv.xz",
                           'url': "https://drive.google.com/uc?id=1vQvmDKoByh5vfKoxTPgjrgDLmZliFU0I&confirm=t",
                           'size': 3775676256,
                           'sha256': "8421d973b4eb119f0739506a0b20ba9508356df97d4673e1c170e871cd134983"},
                      '2D_mold2':
                          {'name': "05.5_combined_2D_moldescs_mold2.tsv.xz",
                           'url': "https://drive.google.com/uc?id=1J65aug2hSUCKv-vCUZZ-2BeB05E6ibMD&confirm=t",
                           'size': 1553510028,
                           'sha256': "0fd1c2b3869c5fa749c21ddd70c5dff621974eccafb8e04fd6f95f3b37242058"},
                      'proteins':
                          {'name': "05.5_combined_set_protein_targets.tsv.xz",
                           'url': "https://drive.google.com/uc?id=1A7XDndNywq_g6g0gdy3LR8PGvjXgYf_Y&confirm=t",
                           'size': 1710756,
                           'sha256': "d8f2cbee8b9849f7c3664fe7e8165c5abf785d374c36a8f151a6ec38fd582d80"},
                      'proteins_unirep':
                          {'name': "05.5_combined_prot_embeddings_unirep.tsv.xz",
                           'url': "https://drive.google.com/uc?id=1Bh6yxiwUZK06dHwmEB6jVcZyc2E1ZJC3&confirm=t",
                           'size': 128869632,
                           'sha256': "9f1fce00e77563481eafc44405f9dc8188d5669ed93cafaee256c0208ca135b8"}
                      }
             }
    # Handle exceptions
    available_versions = list(files.keys())
    if isinstance(version, list):
        for _version in version:
            if _version not in available_versions + ['latest', 'all']:
                raise ValueError(f'version can only be one of [{", ".join(["latest"] + available_versions)}]')
    # Identify version
    latest_version = sorted(available_versions, key=lambda s: [int(u) for u in s.split('.')])[-1]
    if version == 'latest':
        version = latest_version
        if progress:
            print(f'Latest version: {version}')
    elif (isinstance(version, list) and 'latest' in version):
        for i in range(len(version)):
            if version[i] == 'latest':
                version[i] = latest_version
    elif version == 'all' or (isinstance(version, list) and 'all' in version):
        version = available_versions
    # Transform to list
    if not isinstance(version, list):
        version = [version]
    if not isinstance(descriptors, list):
        descriptors = [descriptors]
    # Remove duplicates of versions
    version = sorted(set(version), key=lambda s: [int(u) for u in s.split('.')])
    # Define root dir for downloads
    if outdir is not None:
        os.environ['PYSTOW_HOME'] = os.path.abspath(outdir)
    papyrus_root = pystow.module('papyrus')
    for _version in version:
        papyrus_version_root = pystow.module('papyrus', _version)
        # Prepare files to be downloaded
        downloads = set()
        downloads.add('data_types')
        downloads.add('data_size')
        downloads.add('readme')
        downloads.add('license')
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
        # Determine total download size
        total = sum(files[_version][ftype]['size'] for ftype in downloads)
        if progress:
            print(f'Number of files to be donwloaded: {len(downloads)}\n'
                  f'Total size: {tqdm.format_sizeof(total)}B')
        # Verify enough disk space
        if not enough_disk_space(papyrus_version_root.base.as_posix(), total, disk_margin):
            print(f'Not enough disk space ({disk_margin:.0%} kept for safety)\n'
                  f'Available: {tqdm.format_sizeof(get_disk_space(papyrus_version_root.base.as_posix()))}B\n'
                  f'Required: {tqdm.format_sizeof(total)}B')
            return
        # Download files
        if progress:
            pbar = tqdm(total=total, desc=f'Donwloading version {_version}', unit='B', unit_scale=True)
        for ftype in downloads:
            download = files[_version][ftype]
            dname, durl, dsize, dhash = download['name'], download['url'], download['size'], download['sha256']
            # Determine path
            if ftype in ['2D_papyrus', '3D_papyrus', 'proteins', 'data_types', 'data_size', 'readme', 'license']:
                fpath = papyrus_version_root.join(name=dname).as_posix()
            elif ftype in ['2D_structures', '3D_structures']:
                fpath = papyrus_version_root.join('structures', name=dname).as_posix()
            else:
                fpath = papyrus_version_root.join('descriptors', name=dname).as_posix()
            # File already exists
            if os.path.isfile(fpath) and assert_sha256sum(fpath, dhash):
                if progress:
                    pbar.update(dsize)
                continue # skip
            # Download file
            correct = False  # ensure file is not corrupted
            retries = RETRIES
            while not correct and retries > 0:  # Allow 3 failures
                session = requests.session()
                res = session.get(durl, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) "
                                                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                                                               "Chrome/39.0.2171.95 "
                                                               "Safari/537.36"},
                                  stream=True, verify=True)
                with open(fpath, 'wb') as fh:
                    for chunk in res.iter_content(chunk_size=CHUNKSIZE):
                        fh.write(chunk)
                        if progress:
                            pbar.update(len(chunk))
                correct = assert_sha256sum(fpath, dhash)
                if not correct:
                    retries -= 1
                    if progress:
                        if retries > 0:
                            message = f'SHA256 hash unexpected for {dname}. Remaining download attempts: {retries}'
                        else:
                            message = f'SHA256 hash unexpected for {dname}. All {RETRIES} attempts failed.'
                        pbar.write(message)
                    os.remove(fpath)
            if retries == 0:
                if progress:
                    pbar.close()
                raise IOError(f'Donwload failed for {dname}')
        if progress:
            pbar.close()
        # Save version number
        json_file = papyrus_root.join(name='versions.json').as_posix()
        if os.path.isfile(json_file):
            data = read_jsonfile(json_file)
            data.append(_version)
            data = sorted(set(data))
            write_jsonfile(data, json_file)
        else:
            write_jsonfile([_version], json_file)
