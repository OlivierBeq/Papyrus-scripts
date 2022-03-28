# -*- coding: utf-8 -*-

import sys
import click

from .download import download_papyrus
from .matchRCSB import get_matches, update_rcsb_data
from .reader import read_papyrus
from .utils.IO import get_num_rows_in_file
from .subsim_search import FPSubSim2
from .fingerprint import Fingerprint, get_fp_from_name

@click.group()
def main():
    pass

@main.command(help='Download Papyrus data.')
@click.option('-o', '--out_dir', 'output_directory', type=str, required=False,
              default=None, nargs=1, show_default=True, metavar='OUTDIR',
              help='Directory where Papyrus data will be stored\n(default: pystow\'s home folder).')
@click.option('--version', '-V', 'version', type=str, required=False, default='latest', nargs=1,
              metavar='XX.X', help='Version of the Papyrus data to be downloaded.')
@click.option('-s', '--stereo', 'stereo', type=click.Choice(['without', 'with', 'both']), required=False,
              default='without', nargs=1, show_default=True,
              help=('Type of data to be downloaded: without (standardised data without stereochemistry), '
                    'with (non-standardised data with stereochemistry), '
                    'both (both standardised and non-standardised data)'))
@click.option('-S', '--structures', 'structs', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Should structures be downloaded (SD file).')
@click.option('-d', '--descriptors', 'descs', type=click.Choice(['mold2', 'cddd', 'mordred', 'fingerprint',
                                                                   'unirep', 'all', 'none']),
              required=False, default=['all'], nargs=1,
              show_default=True, multiple=True,
              help=('Type of descriptors to be downloaded: mold2 (777 2D Mold2 descriptors), '
                    'cddd: (512 2D continuous data-driven descriptors), '
                    'mordred: (1613 2D or 1826 3D mordred descriptors) ,\n'
                    'fingerprint (2048 bits 2D RDKit Morgan fingerprint with radius 3 '
                    'or 2048 bits extended 3-dimensional fingerprints of level 5), '
                    'unirep (6660 UniRep deep-learning protein sequence representations '
                    'containing 64, 256 and 1900-bit average hidden states, '
                    'final hidden states and final cell states), '
                    'all (all descriptors for the selected stereochemistry), or '
                    'none (do not download any descriptor).'))
def download(output_directory, version, stereo, structs, descs):
    download_papyrus(outdir=output_directory,
                     version=version,
                     nostereo=stereo in ['without', 'both'],
                     stereo=stereo in ['with', 'both'],
                     structures=structs,
                     descriptors=descs,
                     progress=True)


@main.command(help='Identify matches of the RCSB PDB data in the Papyrus data.')
@click.option('--indir', '-i', 'indir', type=str, required=False, default=None, nargs=1,
              metavar='INDIR', show_default=True,
              help='Directory where Papyrus data will be stored\n(default: pystow\'s home folder).')
@click.option('--output', '-o', 'output', type=str, required=True, default=None, nargs=1,
              metavar='OUTFILE', help='Output file containing the PDB matched Papytus data.')
@click.option('--version', '-V', 'version', type=str, required=False, default='latest', nargs=1,
              metavar='XX.X', help='Version of the Papyrus data to be mapped (default: latest).')
@click.option('-3D', 'is3D', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Toggle matching the non-standardized 3D data.')
@click.option('-O', '--overwrite', 'overwrite', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Toggle overwriting recently downloaded cache files.')
@click.option('--verbose', 'verbose', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Display progress.')
def pdbmatch(indir, output, version, is3D, overwrite, verbose):
    CHUNKSIZE = 1000000
    update_rcsb_data(root_folder=indir, overwrite=overwrite, verbose=verbose)
    data = read_papyrus(is3d=is3D, version=version, chunksize=CHUNKSIZE, source_path=indir)
    total = get_num_rows_in_file('bioactivities', is3D=is3D, version=version, root_folder=indir)
    matched_data = get_matches(data=data, root_folder=indir, verbose=verbose,
                               total=int(round(total / CHUNKSIZE, 0)))
    for i, chunk in enumerate(matched_data):
        if i == 0:
            # Create the output file
            chunk.to_csv(output, sep='\t', index=False)
        else:
            # Append to the output file
            chunk.to_csv(output, sep='\t', index=False, header=False, mode='a')


@main.command(help='Create a FPSubSim2 library substructure/similarity searches.')
@click.option('-i, --indir', 'indir', type=str, required=False, default=None, nargs=1,
              metavar='INDIR', show_default=True,
              help='Directory where Papyrus data will be stored\n(default: pystow\'s home folder).')
@click.option('-o', '--output', 'output', type=str, required=True, default=None, nargs=1, metavar='OUTFILE',
              help='Output FPSubSim2 file (default: create a file in the current directory).')
@click.option('--version', '-V', 'version', type=str, required=False, default='latest', nargs=1,
              metavar='XX.X', help='Version of the Papyrus data to be mapped (default: latest).')
@click.option('-3D', 'is3D', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Toggle matching the non-standardized 3D data.')
@click.option('-F', '--fingerprint', 'fingerprint', type=str, required=False, default=None, multiple=True,
              metavar='FPname;param1=value1;param2=value2;...',
              help='Fingerprints with paprameters to be calculated for similarity searches.')
@click.option('--verbose', 'verbose', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Display progress.')
@click.option('--njobs', 'njobs', type=int, required=False, default=1, nargs=1, show_default=True,
              help='Number of concurrent processes (default: 1).')
def fpsubsim2(indir, output, version, is3D, fingerprint, verbose, njobs):
    fpss = FPSubSim2()
    if len(fingerprint) == 0:
        fpss.create_from_papyrus(is3d=is3D, version=version, outfile=output, fingerprint=None, root_folder=indir,
                                 progress=verbose, njobs=njobs)
    else:
        # Obtain available fingerprint names and parameter names
        fp_correct_values = {}
        for fp_class in Fingerprint.derived():
            fp = fp_class()
            fp_correct_values[fp.name] = fp.params
        # Parse fingerprints signatures
        # Must be in the format FPname;parameter1=value1;parameter2=value2
        fingerprints = []
        for fp in fingerprint:
            fp_param = fp.split(';')
            # Obtain fingerprint name
            fp_name = fp_param.pop(0)
            # Check fingerprint type is available
            if fp_name not in fp_correct_values.keys():
                print(f'Fingerprint must be one of {", ".join(fp_correct_values.keys())}')
                sys.exit()
            # Convert parameters to dict
            fp_param = dict(param.split('=') for param in fp_param)
            # Check all parameter names are legitimate
            for param_name, param_value in fp_param.items():
                if param_name not in fp_correct_values[fp_name].keys():
                    print(f'Parameters for fingerprint {fp_name} '
                          f'are {", ".join(fp_correct_values[fp_name].keys())}')
                # Convert type of parameter values
                try:
                    fp_param[param_name] = eval(param_value)
                except:
                    print(f'Parameters were wrongly formatted: {param_value}')
                    sys.exit()
            fingerprints.append(get_fp_from_name(fp_name, **fp_param))
        fpss.create_from_papyrus(is3d=is3D, version=version, outfile=output, fingerprint=fingerprints, root_folder=indir,
                                 progress=verbose, njobs=njobs)