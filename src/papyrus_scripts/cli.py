# -*- coding: utf-8 -*-

"""Command line interface of the Papyrus-scripts."""

import ast
import inspect
import os
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
def main() -> None:
    """Group allowing subcommands to be defined."""
    pass


@main.command(help='Download Papyrus data.', context_settings=CONTEXT_SETTINGS)
@click.option('-o', '--out_dir', 'output_directory', type=str, required=False,
              default=None, nargs=1, show_default=True, metavar='OUTDIR',
              help="Directory where Papyrus data will be stored\n(default: pystow's home folder).",
              )
@click.option('--version', '-V', 'version', required=False, default=['latest'], multiple=True,
              metavar='YYYY.MM[.R]', help='Version of the Papyrus data to be downloaded (can also be "all").',
              )
@click.option('--more', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Should other data than Papyrus++ be downloaded '
                                      '(considered only when --stereo is "without" or "both").',
              )
@click.option('-s', '--stereo', 'stereo', type=click.Choice(['without', 'with', 'both']),
              required=False, default='without', nargs=1, show_default=True,
              help='Type of data to be downloaded.',
              )
@click.option('-S', '--structures', 'structs', is_flag=True, required=False, default=False,
              nargs=1, show_default=True, help='Should structures be downloaded (SD file).',
              )
@click.option('-d', '--descriptors', 'descs',
              type=click.Choice(['mold2', 'cddd', 'mordred', 'fingerprint',
                                 'unirep', 'prodec', 'all', 'none'],
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
                    "'none' (do not download any descriptor)."),
              )
@click.option('--force', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Force download if disk space is low.',
              )
@click.option('--all-revisions', 'all_revisions', is_flag=True, required=False, default=False,
              show_default=True,
              help='When set, "all" and two-part version aliases expand to every known '
                   'revision rather than only the latest revision per version.',
              )
@click.option('--keep-xz', 'keep_xz', is_flag=True, required=False, default=False,
              show_default=True,
              help='Keep downloaded .xz files as-is instead of converting them to Parquet '
                   '(and deleting the .xz originals). Needed if you intend to transform '
                   'compression with the "convert" command.',
              )
def download(output_directory: str | None, version: tuple[str, ...] | list[str], more: bool, stereo: str,
            structs: bool, descs: tuple[str, ...] | list[str], force: bool, all_revisions: bool,
            keep_xz: bool) -> None:
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
        keep_xz=keep_xz,
    )


@main.command(help='Remove Papyrus data.', context_settings=CONTEXT_SETTINGS)
@click.option('-o', '--out_dir', 'output_directory', type=str, required=False,
              default=None, nargs=1, show_default=True, metavar='OUTDIR',
              help="Directory where Papyrus data will be removed\n(default: pystow's home folder).",
              )
@click.option('--version', '-V', 'version', required=False, default=['latest'], multiple=True,
              metavar='YYYY.MM[.R]', help='Version of the Papyrus data to be removed.',
              )
@click.option('--papyruspp', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Should Papyrus++ bioactivities be removed.',
              )
@click.option('-s', '--stereo', 'stereo', type=click.Choice(['without', 'with', 'both']),
              required=False, default='without', nargs=1, show_default=True,
              help='Type of data to be removed.',
              )
@click.option('-B', '--bioactivities', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Should bioactivities be removed (TSV file).',
              )
@click.option('-P', '--proteins', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Should protein data be removed (TSV file).',
              )
@click.option('-S', '--structures', 'structs', is_flag=True, required=False, default=False,
              nargs=1, show_default=True, help='Should structures be removed (SD file).',
              )
@click.option('-d', '--descriptors', 'descs',
              type=click.Choice(['mold2', 'cddd', 'mordred', 'fingerprint',
                                 'unirep', 'prodec', 'all', 'none'],
                                ),
              required=False, default=['none'], nargs=1, show_default=True, multiple=True,
              help='Type of descriptors to be removed.',
              )
@click.option('-O', '--other_files', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Should other files be removed (e.g. LICENSE, README).',
              )
@click.option('--remove_version', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Should the given Papyrus version(s) be removed.',
              )
@click.option('--remove_root', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Should all Papyrus data and versions be removed.',
              )
@click.option('--force', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Skip confirmation when removing the root directory.',
              )
@click.option('--all-revisions', 'all_revisions', is_flag=True, required=False, default=False,
              show_default=True,
              help='When set, "all" and two-part version aliases expand to every known '
                   'revision rather than only the latest revision per version.',
              )
def clean(output_directory: str | None, version: tuple[str, ...] | list[str], papyruspp: bool, stereo: str,
         bioactivities: bool, proteins: bool, structs: bool, descs: tuple[str, ...] | list[str],
         other_files: bool, remove_version: bool, remove_root: bool, force: bool,
         all_revisions: bool) -> None:
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
              context_settings=CONTEXT_SETTINGS,
              )
@click.option('--indir', '-i', 'indir', type=str, required=False, default=None, nargs=1,
              metavar='INDIR', show_default=True,
              help="Directory where Papyrus data is stored\n(default: pystow's home folder).",
              )
@click.option('--output', '-o', 'output', type=str, required=True, default=None, nargs=1,
              metavar='OUTFILE', help='Output file containing the PDB-matched Papyrus data.',
              )
@click.option('--version', '-V', 'version', type=str, required=False, default='latest', nargs=1,
              metavar='YYYY.MM[.R]', help='Version of the Papyrus data to be mapped (default: latest).',
              )
@click.option('--more', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Should other data than Papyrus++ be included.',
              )
@click.option('-3D', 'is3D', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Toggle matching the non-standardized 3D data.',
              )
@click.option('-O', '--overwrite', 'overwrite', is_flag=True, required=False, default=False,
              nargs=1, show_default=True, help='Toggle overwriting recently downloaded cache files.',
              )
@click.option('--verbose', 'verbose', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Display progress.',
              )
def pdbmatch(indir: str | None, output: str, version: str, more: bool, is3D: bool,
            overwrite: bool, verbose: bool) -> None:
    """CLI to match Papyrus data against RCSB PDB structures."""
    CHUNKSIZE = 1_000_000
    update_rcsb_data(root_folder=indir, overwrite=overwrite, verbose=verbose)
    data = read_papyrus(is3d=is3D, version=version, plusplus=not more,
                        chunksize=CHUNKSIZE, source_path=indir,
                        )
    total = get_num_rows_in_file('bioactivities', is3D=is3D, version=version,
                                  plusplus=not more, root_folder=indir)
    matched_data = get_matches(
        data=data, root_folder=indir, verbose=verbose,
        total=int(round(total / CHUNKSIZE, 0)), update=False,
    )
    output_path = Path(output)
    tmp_path = output_path.with_name(f'{output_path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp')
    for stale in output_path.parent.glob(f'{output_path.name}.*.tmp'):
        stale.unlink(missing_ok=True)
    wrote_any = False
    try:
        for i, chunk in enumerate(matched_data):
            chunk.to_csv(tmp_path, sep='\t', index=False, header=(i == 0),
                         mode='w' if i == 0 else 'a',
                         )
            wrote_any = True
        if wrote_any:
            tmp_path.replace(output_path)
    finally:
        tmp_path.unlink(missing_ok=True)


class Mutex(click.Option):
    """A click.Option that becomes required only if none of `not_required_if` are set."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Configure a click.Option required unless one of `not_required_if` is set.

        Derived from: https://stackoverflow.com/a/61684480
        """
        self.not_required_if: list = kwargs.pop("not_required_if")
        if not self.not_required_if:
            raise ValueError("'not_required_if' parameter required")
        if not isinstance(self.not_required_if, list):
            raise TypeError("'not_required_if' must be a list")
        kwargs["help"] = (
                kwargs.get("help", "")
                + ' NOTE: This argument is mutually exclusive with '
                + ", ".join(self.not_required_if) + "."
        ).strip()
        super().__init__(*args, **kwargs)

    def handle_parse_result(
            self,
            ctx: click.Context,
            opts: Mapping[str, Any],
            args: list[str],
    ) -> tuple[Any, list[str]]:
        """Enforce mutual exclusivity with `not_required_if` before delegating to click."""
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
                        f"{other_param.human_readable_name}.",
                    )
                elif other_opt:
                    self.required = False
        return super().handle_parse_result(ctx, opts, args)


def _versioned_outfile(output: str | None, version_: str, multi: bool) -> str | None:
    """Suffix *output* with *version_* when *multi* versions are requested.

    Without this, every version would overwrite the same *output* path.
    """
    if output is None or not multi:
        return output
    path = Path(output)
    return str(path.with_name(f'{path.stem}_{version_}{path.suffix}'))


@main.command(help='Create a FPSubSim2 library for substructure/similarity searches.',
              context_settings=CONTEXT_SETTINGS,
              )
@click.option('-i, --indir', 'indir', type=str, required=False, default=None, nargs=1,
              metavar='INDIR', show_default=True,
              help="Directory where Papyrus data is stored\n(default: pystow's home folder).",
              )
@click.option('-o', '--output', 'output', type=str, default=None, nargs=1, metavar='OUTFILE',
              required=True, cls=Mutex, not_required_if=['fhelp'],
              help='Output FPSubSim2 file.',
              )
@click.option('--version', '-V', 'version', type=str, required=False, default=['latest'],
              multiple=True, metavar='YYYY.MM[.R]',
              help='Version of the Papyrus data to be mapped (default: latest).',
              )
@click.option('-3D', 'is3D', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Toggle matching the non-standardized 3D data.',
              )
@click.option('--verbose', 'verbose', is_flag=True, required=False, default=False, nargs=1,
              show_default=True, help='Display progress.',
              )
@click.option('--njobs', 'njobs', type=int, required=False, default=1, nargs=1,
              show_default=True, help='Number of concurrent processes (default: 1).',
              )
@click.option('-F', '--fingerprint', 'fingerprint', type=str, required=False,
              default=['Morgan'], multiple=True,
              metavar='FPname[;param1=value1[;param2=value2[;...]]]',
              help='Fingerprints to be calculated for similarity searches.',
              )
@click.option('--fhelp', 'fingerprint_help', is_flag=True, default=False, required=False,
              help='Show advanced help about fingerprints.',
              )
def fpsubsim2(indir: str | None, output: str | None, version: tuple[str, ...], is3D: bool,
             fingerprint: tuple[str, ...], verbose: bool, njobs: int, fingerprint_help: bool) -> None:
    """CLI to create a database for similarity and substructure searches."""
    if fingerprint_help:
        fp_name_list, fp_no_parameter_list, fp_parameter_list = [], [], []
        # Fingerprint (unlike a leaf subclass) always has subclasses, so
        # derived() always returns a list here. Every concrete subclass
        # overrides __init__ with its own no-argument signature; mypy only
        # sees the abstract base's.
        for fp_type in Fingerprint.derived():  # type: ignore[union-attr]
            try:
                # Skip fingerprints whose optional dependencies (openbabel,
                # FPSim2) are not installed, matching get_fp_from_name.
                fp_name = fp_type().name  # type: ignore[call-arg]
            except ImportError:
                continue
            fp_params = [
                (key, value.default)
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
            '  FPname:\n' + '\n'.join(fp_name_list) + '\n\n'
                                                       '  Fingerprints without parameters:\n' + '\n'.join(
                fp_no_parameter_list,
            ) + '\n\n'
                "  Other fingerprints' parameter names and default values:\n" + '\n'.join(fp_parameter_list),
        )
        sys.exit()

    # `output` is only allowed to be None via the Mutex/fhelp path above,
    # already handled (and exited) by this point.
    if output is None:
        raise RuntimeError('output is None despite the fhelp/Mutex path exiting earlier')
    if output.lower() == 'none':
        output = None

    fpss = FPSubSim2()
    multi_version = len(version) > 1
    if 'none' in [fp.lower() for fp in fingerprint]:
        for version_ in version:
            fpss.create_from_papyrus(is3d=is3D, version=version_,
                                     outfile=_versioned_outfile(output, version_, multi_version),
                                     fingerprint=None, root_folder=indir,
                                     progress=verbose, njobs=njobs,
                                     )
    else:
        # Fingerprint (unlike a leaf subclass) always has subclasses, so
        # derived() always returns a list here. Every concrete subclass
        # overrides __init__ with its own no-argument signature; mypy only
        # sees the abstract base's.
        fp_correct_values = {}
        for fp_class in Fingerprint.derived():  # type: ignore[union-attr]
            try:
                # Skip fingerprints whose optional dependencies (openbabel,
                # FPSim2) are not installed, matching get_fp_from_name.
                fp_instance = fp_class()  # type: ignore[call-arg]
            except ImportError:
                continue
            fp_correct_values[fp_instance.name] = fp_instance.params
        fingerprints = []
        for fp in fingerprint:
            fp_param_list = fp.split(';')
            fp_name = fp_param_list.pop(0)
            if fp_name not in fp_correct_values:
                print(f'Fingerprint must be one of {", ".join(fp_correct_values)}')
                sys.exit()
            fp_param_values: dict[str, Any] = dict(param.split('=') for param in fp_param_list)
            for param_name, param_value in fp_param_values.items():
                if param_name not in fp_correct_values[fp_name]:
                    print(f'Parameters for fingerprint {fp_name} '
                          f'are {", ".join(fp_correct_values[fp_name])}',
                          )
                try:
                    fp_param_values[param_name] = ast.literal_eval(param_value)
                except (ValueError, SyntaxError) as e:
                    print(f'Parameter {param_name!r} for fingerprint {fp_name!r} is not a '
                         f'valid Python literal: {param_value!r} ({e})')
                    sys.exit()
            fingerprints.append(get_fp_from_name(fp_name, **fp_param_values))
        for version_ in version:
            fpss.create_from_papyrus(is3d=is3D, version=version_,
                                     outfile=_versioned_outfile(output, version_, multi_version),
                                     fingerprint=fingerprints, root_folder=indir,
                                     progress=verbose, njobs=njobs,
                                     )


@main.command(
    help='Transform the compression of Papyrus files from LZMA to Gzip and vice-versa.',
    context_settings=CONTEXT_SETTINGS,
)
@click.option('-i', '--indir', 'indir', type=str, required=False, default=None, nargs=1,
              metavar='INDIR', show_default=True,
              help="Directory where Papyrus data is stored\n(default: pystow's home folder).",
              )
@click.option('-v', '--version', 'version', type=str, required=False, default='latest',
              multiple=False, metavar='YYYY.MM[.R]',
              help='Version of the Papyrus data to be transformed (default: latest).',
              )
@click.option('-f', '--format', 'format', type=click.Choice(['xz', 'gzip']),
              required=False, default=None, nargs=1, show_default=True, multiple=False,
              help='Compression type to transform the data to. Inferred if not specified.',
              )
@click.option('-l', '--level', 'level', type=click.IntRange(0, 9),
              required=False, default=None, nargs=1, show_default=True, multiple=False,
              help='Compression level of output files.',
              )
@click.option('-e', '--extreme', 'extreme', is_flag=True, required=False, default=False,
              nargs=1, show_default=True, help='Toggle extreme compression.',
              )
def convert(indir: str | None, version: str, format: str | None,
           level: int | None, extreme: bool) -> None:
    """CLI to interconvert Papyrus data between GZIP and XZ compression."""
    if isinstance(version, tuple):  # pragma: no cover - version is multiple=False here, always a str
        version = list(version)
    # resolve version to a PapyrusVersion, then use its on-disk folder name
    pv = process_data_version(version, indir)
    version_dir = papyrus_version_module(pv, indir).base

    xz_files = [f for f in version_dir.rglob('*') if f.is_file() and f.name.lower().endswith('.xz')]
    gz_files = [f for f in version_dir.rglob('*') if f.is_file() and f.name.lower().endswith('.gz')]

    # Converting to Gzip needs the .xz originals - download_papyrus() deletes
    # them by default after converting tabular files to Parquet, so their
    # absence alongside .parquet files means they must be re-fetched rather
    # than simply being missing/not-yet-downloaded.
    if not xz_files and format in (None, 'gzip'):
        parquet_files = [f for f in version_dir.rglob('*.parquet') if f.is_file()]
        if parquet_files:
            raise FileNotFoundError(
                f'No .xz files found in {version_dir}, but {len(parquet_files)} .parquet '
                f'file(s) are present: download_papyrus() already converted the tabular '
                f'.xz files to Parquet and deleted the originals. Re-download with '
                f"download_papyrus(..., keep_xz=True) (or the CLI's "
                f'`papyrus download --keep-xz` flag) to retain the .xz files needed here.',
            )

    if format is None:
        if len(xz_files) > len(gz_files):
            format = 'gzip'
        elif gz_files:
            format = 'xz'
        else:
            raise ValueError(
                'Equal number of LZMA and GZIP files — please specify the output format.',
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
                             extreme=extreme, progress=True,
                             )
            filepath.unlink()


@main.command(
    help='Run extensive reader.py tests against real, locally downloaded Papyrus data '
         '(network/I-O heavy - opt-in, not part of the routine test suite).',
    context_settings=CONTEXT_SETTINGS,
)
@click.option('-V', '--version', 'version', type=str, required=False, default='latest', nargs=1,
              metavar='YYYY.MM[.R]', help='Version of the Papyrus data to test against (default: latest).',
              )
@click.option('-i', '--indir', 'indir', type=str, required=False, default=None, nargs=1,
              metavar='INDIR', show_default=True,
              help="Directory where Papyrus data is stored\n(default: pystow's home folder).",
              )
@click.option('--download', 'download', is_flag=True, required=False, default=False,
              help='Download the version first (bioactivities, protein targets, structures, '
                   'both stereo variants, all descriptors) if not already present locally.',
              )
@click.option('--sample-size', 'sample_size', type=int, required=False, default=25, show_default=True,
              help='Rows/molecules sampled per bounded check, keeping large-file checks fast '
                   'regardless of the true file size.',
              )
@click.option('-v', '--verbose', 'verbose', is_flag=True, required=False, default=False,
              help='Verbose pytest output.',
              )
def test_real_data(version: str, indir: str | None, download: bool, sample_size: int, verbose: bool) -> None:
    """CLI to run extensive reader.py tests against real, locally downloaded Papyrus data."""
    try:
        import pytest
    except ImportError as err:
        raise click.UsageError(
            "pytest is required for this command: pip install 'papyrus-scripts[testing]'",
        ) from err

    test_file = Path(__file__).resolve().parents[2] / 'tests' / 'test_reader_real_data.py'
    if not test_file.is_file():
        raise click.UsageError(
            f'{test_file} not found - this command only works from a full source checkout '
            'of the papyrus-scripts repository (test files are not packaged for distribution).',
        )

    if download:
        download_papyrus(
            outdir=indir, version=version,
            nostereo=True, stereo=True, only_pp=False,
            structures=True, descriptors='all',
            progress=True,
        )

    pv = process_data_version(version, indir)

    os.environ['PAPYRUS_REAL_DATA_VERSION'] = pv.pystow_path_key
    if indir is not None:
        os.environ['PAPYRUS_REAL_DATA_ROOT'] = str(indir)
    os.environ['PAPYRUS_REAL_DATA_SAMPLE_SIZE'] = str(sample_size)

    args = [str(test_file), '-v'] if verbose else [str(test_file)]
    sys.exit(pytest.main(args))
