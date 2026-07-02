# -*- coding: utf-8 -*-

"""Command line interface of the Papyrus-scripts"""

import ast
import inspect
import sys
from collections import defaultdict

import click

from .download import download_papyrus, remove_papyrus
from .fingerprint import Fingerprint, get_fp_from_name
from .matchRCSB import get_matches, update_rcsb_data
from .reader import read_papyrus
from .subsim_search import FPSubSim2
from .utils.IO import (
    convert_gz_to_xz,
    convert_xz_to_gz,
    get_num_rows_in_file,
    papyrus_version_module,
    process_data_version,
)

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    """Group allowing subcommands to be defined"""
    pass


@main.command(help='Download Papyrus data.', context_settings=CONTEXT_SETTINGS)
@click.option('-o', '--out_dir', 'output_directory', type=str, required=False,
              default=None, nargs=1, show_default=True, metavar='OUTDIR',
              help="Directory where Papyrus data will be stored\n(default: pystow's home folder)."
              )
@click.option('--version', '-V', 'version', required=False, default=['latest'], multiple=True,
              metavar='YYYY.MM[.R]', help='Version of the Papyrus data to be downloaded (can also be "all").'
              )
@click.option('--more', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Should other data than Papyrus++ be downloaded '
                                      '(considered only when --stereo is "without" or "both").'
              )
@click.option('-s', '--stereo', 'stereo', type=click.Choice(['without', 'with', 'both']),
              required=False, default='without', nargs=1, show_default=True,
              help='Type of data to be downloaded.'
              )
@click.option('-S', '--structures', 'structs', is_flag=True, required=False, default=False,
              nargs=1, show_default=True, help='Should structures be downloaded (SD file).'
              )
@click.option('-d', '--descriptors', 'descs',
              type=click.Choice(['mold2', 'cddd', 'mordred', 'fingerprint',
                                 'unirep', 'prodec', 'all', 'none']
                                ),
              required=False, default=['none'], nargs=1, show_default=True, multiple=True,
              help=("Type of descriptors to download. 'mold2' (777 2D Mold2 descriptors), "
                    "'cddd': (512 2D continuous data-driven descriptors), "
                    "'mordred': (1613 2D or 1826 3D mordred descriptors) ,\n"
                    "'fingerprint' (2048 bits 2D RDKit Morgan fingerprint with radius 3 "
                    "or 2048 bits extended 3-dimensional fingerprints of level 5), "
                    "'unirep' (6660 UniRep deep-learning protein sequence representations "
                    "containing 64, 256 and 1900-bit average hidden states, "
                    "final hidden states and final cell states), "
                    "'prodec' (all ProDEC descriptors transformed with 50 average domains and lag 20), or "
                    "'all' (all descriptors for the selected stereochemistry), or "
                    "'none' (do not download any descriptor).")
              )
@click.option('--force', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Force download if disk space is low.'
              )
@click.option('--all-revisions', 'all_revisions', is_flag=True, required=False, default=False,
              show_default=True,
              help='When set, "all" and two-part version aliases expand to every known '
                   'revision rather than only the latest revision per version.'
              )
def download(output_directory, version, more, stereo, structs, descs, force, all_revisions):
    """CLI to download the Papyrus data."""
    if isinstance(version, tuple):
        version = list(version)
    if isinstance(descs, tuple):
        descs = list(descs)
    download_papyrus(
        outdir=output_directory,
        version=version,
        nostereo=stereo in ['without', 'both'],
        stereo=stereo in ['with', 'both'],
        only_pp=not more,
        structures=structs,
        descriptors=descs,
        progress=True,
        disk_margin=0.0 if force else 0.1,
        all_revisions=all_revisions,
    )


@main.command(help='Remove Papyrus data.', context_settings=CONTEXT_SETTINGS)
@click.option('-o', '--out_dir', 'output_directory', type=str, required=False,
              default=None, nargs=1, show_default=True, metavar='OUTDIR',
              help="Directory where Papyrus data will be removed\n(default: pystow's home folder)."
              )
@click.option('--version', '-V', 'version', required=False, default=['latest'], multiple=True,
              metavar='YYYY.MM[.R]', help='Version of the Papyrus data to be removed.'
              )
@click.option('--papyruspp', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Should Papyrus++ bioactivities be removed.'
              )
@click.option('-s', '--stereo', 'stereo', type=click.Choice(['without', 'with', 'both']),
              required=False, default='without', nargs=1, show_default=True,
              help='Type of data to be removed.'
              )
@click.option('-B', '--bioactivities', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Should bioactivities be removed (TSV file).'
              )
@click.option('-P', '--proteins', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Should protein data be removed (TSV file).'
              )
@click.option('-S', '--structures', 'structs', is_flag=True, required=False, default=False,
              nargs=1, show_default=True, help='Should structures be removed (SD file).'
              )
@click.option('-d', '--descriptors', 'descs',
              type=click.Choice(['mold2', 'cddd', 'mordred', 'fingerprint',
                                 'unirep', 'prodec', 'all', 'none']
                                ),
              required=False, default=['none'], nargs=1, show_default=True, multiple=True,
              help='Type of descriptors to be removed.'
              )
@click.option('-O', '--other_files', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Should other files be removed (e.g. LICENSE, README).'
              )
@click.option('--remove_version', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Should the given Papyrus version(s) be removed.'
              )
@click.option('--remove_root', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Should all Papyrus data and versions be removed.'
              )
@click.option('--force', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Skip confirmation when removing the root directory.'
              )
@click.option('--all-revisions', 'all_revisions', is_flag=True, required=False, default=False,
              show_default=True,
              help='When set, "all" and two-part version aliases expand to every known '
                   'revision rather than only the latest revision per version.'
              )
def clean(output_directory, version, papyruspp, stereo, bioactivities, proteins,
          structs, descs, other_files, remove_version, remove_root, force, all_revisions):
    """CLI to remove the Papyrus data."""
    if isinstance(version, tuple):
        version = list(version)
    if isinstance(descs, tuple):
        descs = list(descs)
    remove_papyrus(
        outdir=output_directory,
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
        progress=True,
        all_revisions=all_revisions,
    )


@main.command(help='Identify matches of the RCSB PDB data in the Papyrus data.',
              context_settings=CONTEXT_SETTINGS
              )
@click.option('--indir', '-i', 'indir', type=str, required=False, default=None, nargs=1,
              metavar='INDIR', show_default=True,
              help="Directory where Papyrus data is stored\n(default: pystow's home folder)."
              )
@click.option('--output', '-o', 'output', type=str, required=True, default=None, nargs=1,
              metavar='OUTFILE', help='Output file containing the PDB-matched Papyrus data.'
              )
@click.option('--version', '-V', 'version', type=str, required=False, default='latest', nargs=1,
              metavar='YYYY.MM[.R]', help='Version of the Papyrus data to be mapped (default: latest).'
              )
@click.option('--more', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Should other data than Papyrus++ be included.'
              )
@click.option('-3D', 'is3D', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Toggle matching the non-standardized 3D data.'
              )
@click.option('-O', '--overwrite', 'overwrite', is_flag=True, required=False, default=False,
              nargs=1, show_default=True, help='Toggle overwriting recently downloaded cache files.'
              )
@click.option('--verbose', 'verbose', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Display progress.'
              )
def pdbmatch(indir, output, version, more, is3D, overwrite, verbose):
    """CLI to match Papyrus data against RCSB PDB structures."""
    CHUNKSIZE = 1_000_000
    update_rcsb_data(root_folder=indir, overwrite=overwrite, verbose=verbose)
    data = read_papyrus(is3d=is3D, version=version, plusplus=not more,
                        chunksize=CHUNKSIZE, source_path=indir
                        )
    total = get_num_rows_in_file('bioactivities', is3D=is3D, version=version, root_folder=indir)
    matched_data = get_matches(
        data=data, root_folder=indir, verbose=verbose,
        total=int(round(total / CHUNKSIZE, 0)), update=False,
    )
    for i, chunk in enumerate(matched_data):
        chunk.to_csv(output, sep='\t', index=False, header=(i == 0),
                     mode='w' if i == 0 else 'a'
                     )


class Mutex(click.Option):
    def __init__(self, *args, **kwargs):
        """Custom class allowing click.Options to be required if other
        click.Options are not set.

        Derived from: https://stackoverflow.com/a/61684480
        """
        self.not_required_if: list = kwargs.pop("not_required_if")
        assert self.not_required_if, "'not_required_if' parameter required"
        assert isinstance(self.not_required_if, list), "'not_required_if' must be a list"
        kwargs["help"] = (
                kwargs.get("help", "")
                + ' NOTE: This argument is mutually exclusive with '
                + ", ".join(self.not_required_if) + "."
        ).strip()
        super().__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        current_opt: bool = self.consume_value(ctx, opts)[0]
        for other_param in ctx.command.get_params(ctx):
            if other_param is self:
                continue
            if (
                    other_param.human_readable_name in self.not_required_if
                    or any(opt.lstrip('-') in self.not_required_if for opt in other_param.opts)
                    or any(opt.lstrip('-') in self.not_required_if for opt in other_param.secondary_opts)
            ):
                other_opt: bool = other_param.consume_value(ctx, opts)[0]
                if other_opt and current_opt:
                    raise click.UsageError(
                        f"Illegal usage: '{self.name}' is mutually exclusive with "
                        f"{other_param.human_readable_name}."
                    )
                elif other_opt:
                    self.required = None
        return super().handle_parse_result(ctx, opts, args)


@main.command(help='Create a FPSubSim2 library for substructure/similarity searches.',
              context_settings=CONTEXT_SETTINGS
              )
@click.option('-i, --indir', 'indir', type=str, required=False, default=None, nargs=1,
              metavar='INDIR', show_default=True,
              help="Directory where Papyrus data is stored\n(default: pystow's home folder)."
              )
@click.option('-o', '--output', 'output', type=str, default=None, nargs=1, metavar='OUTFILE',
              required=True, cls=Mutex, not_required_if=['fhelp'],
              help='Output FPSubSim2 file.'
              )
@click.option('--version', '-V', 'version', type=str, required=False, default=['latest'],
              multiple=True, metavar='YYYY.MM[.R]',
              help='Version of the Papyrus data to be mapped (default: latest).'
              )
@click.option('-3D', 'is3D', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Toggle matching the non-standardized 3D data.'
              )
@click.option('--verbose', 'verbose', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Display progress.'
              )
@click.option('--njobs', 'njobs', type=int, required=False, default=1, nargs=1,
              show_default=True, help='Number of concurrent processes (default: 1).'
              )
@click.option('-F', '--fingerprint', 'fingerprint', type=str, required=False,
              default=['Morgan'], multiple=True,
              metavar='FPname[;param1=value1[;param2=value2[;...]]]',
              help='Fingerprints to be calculated for similarity searches.'
              )
@click.option('--fhelp', 'fingerprint_help', is_flag=True, default=False, required=False,
              help='Show advanced help about fingerprints.'
              )
def fpsubsim2(indir, output, version, is3D, fingerprint, verbose, njobs, fingerprint_help):
    """CLI to create a database for similarity and substructure searches."""
    if fingerprint_help:
        fp_name_list, fp_no_parameter_list, fp_parameter_list = [], [], []
        for fp_type in Fingerprint.derived():
            fp_name = fp_type().name
            fp_params = [
                (key, value._default)
                for key, value in inspect.signature(fp_type.__init__).parameters.items()
                if key != 'self'
            ]
            fp_name_list.append(f"    {fp_name}")
            if fp_params:
                fp_parameter_list.append(f"      {fp_name}")
                fp_parameter_list.extend(
                    f"        {pname} = {pdefault}" for pname, pdefault in fp_params
                )
            else:
                fp_no_parameter_list.append(f"      {fp_name}")
        print(
            'Advanced options for FPSubSim2 fingerprints\n\n'
            'Usage: papyrus fpsubsim2 [OPTIONS] [-F FINGERPRINT] [-F FINGERPRINT] ...\n\n'
            'Fingerprint:\n\n'
            '  Fingerprint signatures must have the following format:\n'
            '     FPname[;param1=value1[;param2=value2[;...]]]\n\n'
            f'  FPname:\n' + '\n'.join(fp_name_list) + '\n\n'
                                                       f'  Fingerprints without parameters:\n' + '\n'.join(
                fp_no_parameter_list
            ) + '\n\n'
                f"  Other fingerprints' parameter names and default values:\n" + '\n'.join(fp_parameter_list)
        )
        sys.exit()

    if output.lower() == 'none':
        output = None

    fpss = FPSubSim2()
    if 'none' in [fp.lower() for fp in fingerprint]:
        for version_ in version:
            fpss.create_from_papyrus(is3d=is3D, version=version_, outfile=output,
                                     fingerprint=None, root_folder=indir,
                                     progress=verbose, njobs=njobs
                                     )
    else:
        fp_correct_values = {fp_class().name: fp_class().params for fp_class in Fingerprint.derived()}
        fingerprints = []
        for fp in fingerprint:
            fp_param = fp.split(';')
            fp_name = fp_param.pop(0)
            if fp_name not in fp_correct_values:
                print(f'Fingerprint must be one of {", ".join(fp_correct_values)}')
                sys.exit()
            fp_param = dict(param.split('=') for param in fp_param)
            for param_name, param_value in fp_param.items():
                if param_name not in fp_correct_values[fp_name]:
                    print(f'Parameters for fingerprint {fp_name} '
                          f'are {", ".join(fp_correct_values[fp_name])}'
                          )
                try:
                    fp_param[param_name] = ast.literal_eval(param_value)
                except (ValueError, SyntaxError) as e:
                    print(f'Parameter {param_name!r} for fingerprint {fp_name!r} is not a '
                         f'valid Python literal: {param_value!r} ({e})')
                    sys.exit()
            fingerprints.append(get_fp_from_name(fp_name, **fp_param))
        for version_ in version:
            fpss.create_from_papyrus(is3d=is3D, version=version_, outfile=output,
                                     fingerprint=fingerprints, root_folder=indir,
                                     progress=verbose, njobs=njobs
                                     )


@main.command(
    help='Transform the compression of Papyrus files from LZMA to Gzip and vice-versa.',
    context_settings=CONTEXT_SETTINGS,
)
@click.option('-i', '--indir', 'indir', type=str, required=False, default=None, nargs=1,
              metavar='INDIR', show_default=True,
              help="Directory where Papyrus data is stored\n(default: pystow's home folder)."
              )
@click.option('-v', '--version', 'version', type=str, required=False, default='latest',
              multiple=False, metavar='YYYY.MM[.R]',
              help='Version of the Papyrus data to be transformed (default: latest).'
              )
@click.option('-f', '--format', 'format', type=click.Choice(['xz', 'gzip']),
              required=False, default=None, nargs=1, show_default=True, multiple=False,
              help='Compression type to transform the data to. Inferred if not specified.'
              )
@click.option('-l', '--level', 'level', type=click.IntRange(0, 9),
              required=False, default=None, nargs=1, show_default=True, multiple=False,
              help='Compression level of output files.'
              )
@click.option('-e', '--extreme', 'extreme', is_flag=True, required=False, default=False,
              nargs=1, show_default=True, help='Toggle extreme compression.'
              )
def convert(indir, version, format, level, extreme):
    """CLI to interconvert Papyrus data between GZIP and XZ compression."""
    if isinstance(version, tuple):
        version = list(version)
    # resolve version to a PapyrusVersion, then use its on-disk folder name
    pv = process_data_version(version, indir)
    version_dir = papyrus_version_module(pv, indir).base

    if format is None:
        counts = defaultdict(list)
        for filepath in version_dir.rglob('*'):
            if not filepath.is_file():
                continue
            if filepath.name.lower().endswith('.xz'):
                counts['gzip'].append(filepath)
            elif filepath.name.lower().endswith('.gz'):
                counts['xz'].append(filepath)
        if len(counts['gzip']) > len(counts['xz']):
            format = 'gzip'
        elif counts['xz']:
            format = 'xz'
        else:
            raise ValueError(
                'Equal number of LZMA and GZIP files — please specify the output format.'
            )

    for filepath in version_dir.rglob('*'):
        if not filepath.is_file():
            continue
        if format == 'gzip' and filepath.name.endswith('.xz'):
            out = filepath.with_suffix('.gz')
            convert_xz_to_gz(filepath, out, compression_level=level, progress=True)
            filepath.unlink()
        elif format == 'xz' and filepath.name.endswith('.gz'):
            out = filepath.with_suffix('.xz')
            convert_gz_to_xz(filepath, out, compression_level=level,
                             extreme=extreme, progress=True
                             )
            filepath.unlink()