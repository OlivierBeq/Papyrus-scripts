# -*- coding: utf-8 -*-

"""Command line interface of the Papyrus-scripts"""

import sys
import os
import inspect
from collections import defaultdict

import click
import pystow

from .download import download_papyrus, remove_papyrus
from .matchRCSB import get_matches, update_rcsb_data
from .reader import read_papyrus
from .utils.IO import get_num_rows_in_file, process_data_version, convert_gz_to_xz, convert_xz_to_gz
from .subsim_search import FPSubSim2
from .fingerprint import Fingerprint, get_fp_from_name


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    """Group allowing subcommands to be defined"""
    pass


@main.command(help='Download Papyrus data.', context_settings=CONTEXT_SETTINGS)
@click.option('-o', '--out_dir', 'output_directory', type=str, required=False,
              default=None, nargs=1, show_default=True, metavar='OUTDIR',
              help='Directory where Papyrus data will be stored\n(default: pystow\'s home folder).')
@click.option('--version', '-V', 'version', required=False, default=['latest'], multiple=True,
              metavar='XX.X', help='Version of the Papyrus data to be downloaded (can also be "all").')
@click.option('--more', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Should other data than Papyrus++ be downloaded '
                                      '(considered only when --stereo is "without" or "both").')
@click.option('-s', '--stereo', 'stereo', type=click.Choice(['without', 'with', 'both']), required=False,
              default='without', nargs=1, show_default=True,
              help=('Type of data to be downloaded: without (standardised data without stereochemistry), '
                    'with (non-standardised data with stereochemistry), '
                    'both (both standardised and non-standardised data).'))
@click.option('-S', '--structures', 'structs', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Should structures be downloaded (SD file).')
@click.option('-d', '--descriptors', 'descs', type=click.Choice(['mold2', 'cddd', 'mordred', 'fingerprint',
                                                                 'unirep', 'prodec', 'all', 'none']),
              required=False, default=['none'], nargs=1,
              show_default=True, multiple=True,
              help=('Type of descriptors to be downloaded: mold2 (777 2D Mold2 descriptors), '
                    'cddd: (512 2D continuous data-driven descriptors), '
                    'mordred: (1613 2D or 1826 3D mordred descriptors) ,\n'
                    'fingerprint (2048 bits 2D RDKit Morgan fingerprint with radius 3 '
                    'or 2048 bits extended 3-dimensional fingerprints of level 5), '
                    'unirep (6660 UniRep deep-learning protein sequence representations '
                    'containing 64, 256 and 1900-bit average hidden states, '
                    'final hidden states and final cell states), '
                    'prodec (all ProDEC descriptors transformed with 50 average domains and lag 20), or '
                    'all (all descriptors for the selected stereochemistry), or '
                    'none (do not download any descriptor).'))
@click.option('--force', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Force download if disk space is low'
                                      '(default: False for 10% disk space margin).')
def download(output_directory, version, more, stereo, structs, descs, force):
    """CLI to download the Papyrus datas."""
    if isinstance(version, tuple):
        version = list(version)
    if isinstance(descs, tuple):
        descs = list(descs)
    download_papyrus(outdir=output_directory,
                     version=version,
                     nostereo=stereo in ['without', 'both'],
                     stereo=stereo in ['with', 'both'],
                     only_pp=not more,
                     structures=structs,
                     descriptors=descs,
                     progress=True,
                     disk_margin=0.0 if force else 0.1)


@main.command(help='Remove Papyrus data.', context_settings=CONTEXT_SETTINGS)
@click.option('-o', '--out_dir', 'output_directory', type=str, required=False,
              default=None, nargs=1, show_default=True, metavar='OUTDIR',
              help='Directory where Papyrus data will be removed\n(default: pystow\'s home folder).')
@click.option('--version', '-V', 'version', required=False, default=['latest'], multiple=True,
              metavar='XX.X', help='Version of the Papyrus data to be removed.')
@click.option('--papyruspp', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Should Papyrus++ bioactivities be removed.')
@click.option('-s', '--stereo', 'stereo', type=click.Choice(['without', 'with', 'both']), required=False,
              default='without', nargs=1, show_default=True,
              help=('Type of data to be removed: without (standardised data without stereochemistry), '
                    'with (non-standardised data with stereochemistry), '
                    'both (both standardised and non-standardised data)'))
@click.option('-B', '--bioactivities', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Should bioactivities be removed (TSV file).')
@click.option('-P', '--proteins', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Should bioactivities be removed (TSV file).')
@click.option('-S', '--structures', 'structs', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Should structures be removed (SD file).')
@click.option('-d', '--descriptors', 'descs',
              type=click.Choice(['mold2', 'cddd', 'mordred', 'fingerprint', 'unirep', 'prodec', 'all', 'none']),
              required=False, default=['none'], nargs=1,
              show_default=True, multiple=True,
              help=('Type of descriptors to be removed: mold2 (777 2D Mold2 descriptors), '
                    'cddd: (512 2D continuous data-driven descriptors), '
                    'mordred: (1613 2D or 1826 3D mordred descriptors) ,\n'
                    'fingerprint (2048 bits 2D RDKit Morgan fingerprint with radius 3 '
                    'or 2048 bits extended 3-dimensional fingerprints of level 5), '
                    'unirep (6660 UniRep deep-learning protein sequence representations '
                    'containing 64, 256 and 1900-bit average hidden states, '
                    'final hidden states and final cell states), '
                    'all (all descriptors for the selected stereochemistry), or '
                    'none (do not download any descriptor).'))
@click.option('-O', '--other_files', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Should other files be removed (e.g. LICENSE, README).')
@click.option('--remove_version', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Should the given Papyrus version(s) be removed.')
@click.option('--remove_root', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Should all Papyrus data and versions be removed.')
@click.option('--force', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Skip confirmation when removing the root directory.')
def clean(output_directory, version, papyruspp, stereo, bioactivities, proteins,structs,
          descs, other_files, remove_version, remove_root, force):
    """CLI to remove the Papyrus data."""
    if isinstance(version, tuple):
        version = list(version)
    if isinstance(descs, tuple):
        descs = list(descs)
    remove_papyrus(outdir=output_directory,
                   version=version,
                   papyruspp=papyruspp,
                   bioactivities=bioactivities,
                   proteins=proteins,
                   nostereo=stereo in ['without', 'both'],
                   stereo=stereo in ['with', 'both'],
                   structures=structs,
                   descriptors=descs,
                   other_files=other_files,
                   version_root=remove_version,
                   papyrus_root=remove_root,
                   force=force,
                   progress=True)


@main.command(help='Identify matches of the RCSB PDB data in the Papyrus data.', context_settings=CONTEXT_SETTINGS)
@click.option('--indir', '-i', 'indir', type=str, required=False, default=None, nargs=1,
              metavar='INDIR', show_default=True,
              help='Directory where Papyrus data will be stored\n(default: pystow\'s home folder).')
@click.option('--output', '-o', 'output', type=str, required=True, default=None, nargs=1,
              metavar='OUTFILE', help='Output file containing the PDB matched Papytus data.')
@click.option('--version', '-V', 'version', type=str, required=False, default='latest', nargs=1,
              metavar='XX.X', help='Version of the Papyrus data to be mapped (default: latest).')
@click.option('--more', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Should other data than Papyrus++ be downloaded '
                                      '(considered only when --stereo is "without" or "both").')
@click.option('-3D', 'is3D', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Toggle matching the non-standardized 3D data.')
@click.option('-O', '--overwrite', 'overwrite', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Toggle overwriting recently downloaded cache files.')
@click.option('--verbose', 'verbose', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Display progress.')
def pdbmatch(indir, output, version, more, is3D, overwrite, verbose):
    CHUNKSIZE = 1000000
    update_rcsb_data(root_folder=indir, overwrite=overwrite, verbose=verbose)
    data = read_papyrus(is3d=is3D, version=version, plusplus=not more, chunksize=CHUNKSIZE, source_path=indir)
    total = get_num_rows_in_file('bioactivities', is3D=is3D, version=version, root_folder=indir)
    matched_data = get_matches(data=data, root_folder=indir, verbose=verbose,
                               total=int(round(total / CHUNKSIZE, 0)), update=False)
    for i, chunk in enumerate(matched_data):
        if i == 0:
            # Create the output file
            chunk.to_csv(output, sep='\t', index=False)
        else:
            # Append to the output file
            chunk.to_csv(output, sep='\t', index=False, header=False, mode='a')


class Mutex(click.Option):
    def __init__(self, *args, **kwargs):
        """Custom class allowing click.Options to be
        required if other click.Options are not set.

        Derived from: https://stackoverflow.com/a/61684480
        """
        self.not_required_if: list = kwargs.pop("not_required_if")

        assert self.not_required_if, "'not_required_if' parameter required"
        assert isinstance(self.not_required_if, list), "'not_required_if' mut be a list"
        kwargs["help"] = (kwargs.get("help", "") + ' NOTE: This argument is mutually exclusive with ' + ", ".join(
            self.not_required_if) + ".").strip()
        super(Mutex, self).__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        """Override base method."""
        current_opt: bool = self.consume_value(ctx, opts)[0]  # Obtain set value
        for other_param in ctx.command.get_params(ctx):
            if other_param is self:  # Ignore the current parameter
                continue
            # Other argument's name or declaration in self.not_required_if
            if other_param.human_readable_name in self.not_required_if or any(
                    opt.lstrip('-') in self.not_required_if for opt in other_param.opts) or any(
                    opt.lstrip('-') in self.not_required_if for opt in other_param.secondary_opts):
                # Get value assigned to the other option
                other_opt: bool = other_param.consume_value(ctx, opts)[0]
                if other_opt:
                    if current_opt:
                        raise click.UsageError(
                            "Illegal usage: '" + str(self.name)
                            + "' is mutually exclusive with "
                            + str(other_param.human_readable_name) + "."
                        )
                    else:
                        self.required = None  # Override requirement
        return super(Mutex, self).handle_parse_result(ctx, opts, args)


@main.command(help='Create a FPSubSim2 library for substructure/similarity searches.',
              context_settings=CONTEXT_SETTINGS)
@click.option('-i, --indir', 'indir', type=str, required=False, default=None, nargs=1,
              metavar='INDIR', show_default=True,
              help='Directory where Papyrus data will be stored\n(default: pystow\'s home folder).')
@click.option('-o', '--output', 'output', type=str, default=None, nargs=1, metavar='OUTFILE',
              required=True, cls=Mutex, not_required_if=['fhelp'],
              help='Output FPSubSim2 file. If "None" determine the most '
                   'convenient name and write to the current directory.')
@click.option('--version', '-V', 'version', type=str, required=False, default=['latest'], multiple=True,
              metavar='XX.X', help='Version of the Papyrus data to be mapped (default: latest).')
@click.option('-3D', 'is3D', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Toggle matching the non-standardized 3D data.')
@click.option('--verbose', 'verbose', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Display progress.')
@click.option('--njobs', 'njobs', type=int, required=False, default=1, nargs=1, show_default=True,
              help='Number of concurrent processes (default: 1).')
@click.option('-F', '--fingerprint', 'fingerprint', type=str, required=False, default=['Morgan'], multiple=True,
              metavar='FPname[;param1=value1[;param2=value2[;...]]]',
              help='Fingerprints with parameters to be calculated for similarity searches '
                   '(default: Morgan fingerprint with 2048 bits and radius 2). '
                   'If "None", calculates all available fingerprints. '
                   'Must be the last argument given')
@click.option('--fhelp', 'fingerprint_help', is_flag=True, default=False, required=False,
              help='Show advanced help about fingerprints.')
def fpsubsim2(indir, output, version, is3D, fingerprint, verbose, njobs, fingerprint_help):
    """CLI to create a database for similarity and substructure searches."""
    # Switch to advanced fingerprint help
    if fingerprint_help:
        fp_name_list = []  # Names of avaliable fingerprints
        fp_no_parameter_list = []  # Formatted names of fingerprints having no parameter
        fp_parameter_list = []  # Formatted names and argument names & values of fingerprints other fingperints
        for fp_type in Fingerprint.derived():
            fp_name = fp_type().name  # Obtain fp name
            # Obtain argument names and default values
            fp_params = [(key, value._default)
                         for key, value in inspect.signature(fp_type.__init__).parameters.items()
                         if key != 'self']
            # Format fp names and arguments
            fp_name_list.append(f"    {fp_name}")
            if len(fp_params):  # Fp has arguments
                fp_parameter_list.append(f"      {fp_name}")
                fp_parameter_list.extend(f"        {param_name} = {default_value}"
                                         for param_name, default_value in fp_params)
            else:  # Fp has no argument
                fp_no_parameter_list.append(f"      {fp_name}")
        # Format the entire lists
        fp_name_list = '\n'.join(fp_name_list)
        fp_parameter_list = '\n'.join(fp_parameter_list)
        fp_no_parameter_list = '\n'.join(fp_no_parameter_list)
        print(f'''Advanced options for FPSubSim2 fingerprints

Usage: papyrus fpsubsim2 [OPTIONS] [-F FINGERPRINT] [-F FINGERPRINT] ...

Fingerprint:

  Fingerprint signatures must have the following format:
     FPname[;param1=value1[;param2=value2[;...]]]
     
  FPname:
{fp_name_list}

  Fingerprints without parameters:
{fp_no_parameter_list}

  Other fingerprints' parameter names and default values:
{fp_parameter_list}''')
        sys.exit()
    # Set output to None if specified, default output file
    # will be created in the current directory
    if output.lower() == 'none':
        output = None
    fpss = FPSubSim2()
    # Set fingerprint to None if specified, default to
    # calculating all available fingerprints
    if 'none' in [fp.lower() for fp in fingerprint]:
        for version_ in version:
            fpss.create_from_papyrus(is3d=is3D, version=version_, outfile=output, fingerprint=None, root_folder=indir,
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
                except Exception as e:
                    print(f'Parameters were wrongly formatted: {param_value}')
                    sys.exit()
            fingerprints.append(get_fp_from_name(fp_name, **fp_param))
        for version_ in version:
            fpss.create_from_papyrus(is3d=is3D, version=version_, outfile=output, fingerprint=fingerprints,
                                     root_folder=indir, progress=verbose, njobs=njobs)


@main.command(help='Transform the compression of Papyrus files from LZMA to Gzip and vice-versa.',
              context_settings=CONTEXT_SETTINGS)
@click.option('-i', '--indir', 'indir', type=str, required=False, default=None, nargs=1,
              metavar='INDIR', show_default=True,
              help='Directory where Papyrus data is stored\n(default: pystow\'s home folder).')
@click.option('-v', '--version', 'version', type=str, required=False, default=['latest'], multiple=False,
              metavar='XX.X', help='Version of the Papyrus data to be transformed (default: latest).')
@click.option('-f', '--format', 'format', type=click.Choice(['xz', 'gzip']),
              required=False, default=None, nargs=1, show_default=True, multiple=False,
              help='Compression type to transform the data to. Is inferred if not specified.')
@click.option('-l', '--level', 'level', type=click.IntRange(0, 9),
              required=False, default=None, nargs=1, show_default=True, multiple=False,
              help='Compression level of output files.')
@click.option('-e', '--extreme', 'extreme', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Should extreme compression be toggled on.')
def convert(indir, version, format, level, extreme):
    """CLI to interconvert Papyrus data from/to GZIP from/to XZ."""
    if isinstance(version, tuple):
        version = list(version)
    if indir is None:
        indir = str(pystow.utils.get_base(''))
    version = process_data_version(version, indir)
    if format is None:
        format = defaultdict(list)
        # Infer from name
        for root, _, files in os.walk(os.path.join(indir, 'papyrus', version)):
            for name in files:
                if name.lower().endswith('xz'):
                    format['gzip'].append(os.path.join(root, name))
                elif name.lower().endswith('gz'):
                    format['xz'].append(os.path.join(root, name))
        if len(format['gzip']) > len(format['xz']):
            format = 'gzip'
        elif len(format['xz']) != 0:
            format = 'xz'
        else:
            raise ValueError('Equal number of LZMA and GZIP files, please indicate the output format.')
    # Transform files of the specified format
    for root, _, files in os.walk(os.path.join(indir, 'papyrus', version)):
        for name in files:
            if format == 'gzip' and name.endswith('xz'):
                convert_xz_to_gz(os.path.join(root, name),
                                 os.path.join(root, name).rstrip('xz') + 'gz',
                                 compression_level=level,
                                 progress=True)
                os.remove(os.path.join(root, name))
            elif format == 'xz' and name.endswith('gz'):
                convert_gz_to_xz(os.path.join(root, name),
                                 os.path.join(root, name).rstrip('gz') + 'xz',
                                 compression_level=level,
                                 extreme=extreme,
                                 progress=True)
                os.remove(os.path.join(root, name))
