# -*- coding: utf-8 -*-

"""Extensive tests for qsar()/pcm() against real, pre-filtered Papyrus
bioactivity subsets - several real targets, quality levels and years, not
just synthetic single-target data (see tests/test_modelling.py for the
lightweight smoke tests). Descriptors are mocked (synthetic, keyed to each
fixture's real connectivity/target_id values) and DecisionTree replaces
XGBoost, to keep this fast while still exercising qsar()/pcm()'s real data
handling at realistic scale.

Fixtures are downloaded from OlivierBeq/Papyrus-modelling on first use and
cached under tests/fixtures/ (gitignored) rather than committed, to keep
the repository small.
"""

import csv
import lzma
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import polars as pl
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.papyrus_scripts.modelling import pcm, qsar
from src.papyrus_scripts.utils.IO import new_session

FIXTURES_DIR = Path(__file__).parent / 'fixtures'
_RESULTS_URL = ('https://raw.githubusercontent.com/OlivierBeq/Papyrus-modelling/'
                'main/manuscript/results/{}')
FIXTURES = [
    'results_filtered_CCRs_human_high.txt.xz',
    'results_filtered_SLC6_high.txt.xz',
    'results_filtered_MonoamineR_human_sample.txt.xz',
]
_MONOAMINE_SOURCE = 'results_filtered_MonoamineR_human.txt.xz'
# Both have a genuine Low/Medium/High Quality mix, unlike the *_high
# fixtures above (100% High despite the name) - kept from the source file.
_MONOAMINE_SAMPLE_TARGETS = {'P21917_WT', 'P50406_WT'}


def _download(name: str) -> Path:
    dest = FIXTURES_DIR / name
    if not dest.exists():
        FIXTURES_DIR.mkdir(exist_ok=True)
        response = new_session().get(_RESULTS_URL.format(name), stream=True)
        response.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + '.part')
        with open(tmp, 'wb') as fh:
            for chunk in response.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        tmp.replace(dest)
    return dest


def _build_monoamine_sample() -> Path:
    dest = FIXTURES_DIR / FIXTURES[-1]
    if dest.exists():
        return dest
    # Streamed row-by-row (the source is ~2 GB decompressed) rather than
    # loaded whole, since only 2 of its 36 targets are kept.
    source = _download(_MONOAMINE_SOURCE)
    with lzma.open(source, 'rt', newline='') as fh_in, lzma.open(dest, 'wt', newline='') as fh_out:
        reader = csv.reader(fh_in, delimiter='\t')
        writer = csv.writer(fh_out, delimiter='\t')
        header = next(reader)
        target_col = header.index('target_id')
        writer.writerow(header)
        for row in reader:
            if row[target_col] in _MONOAMINE_SAMPLE_TARGETS:
                writer.writerow(row)
    return dest


def _load_fixture(name: str) -> pl.DataFrame:
    path = _build_monoamine_sample() if name == FIXTURES[-1] else _download(name)
    # infer_schema_length=None: some columns (e.g. type_KD) mix plain ints
    # with ';'-joined strings past polars' default 100-row inference
    # sample, which would otherwise misinfer them as int64.
    with lzma.open(path, 'rb') as fh:
        return pl.read_csv(fh, separator='\t', infer_schema_length=None)


def _synthetic_mol_descriptors(df: pl.DataFrame) -> pl.DataFrame:
    connectivity = df['connectivity'].unique().to_list()
    return pl.DataFrame({
        'connectivity': connectivity,
        'Desc_1': [i / len(connectivity) for i in range(len(connectivity))],
        'Desc_2': [(i * 7 % 13) / 13 for i in range(len(connectivity))],
    })


def _synthetic_prot_descriptors(df: pl.DataFrame) -> pl.DataFrame:
    targets = df['target_id'].unique().to_list()
    return pl.DataFrame({
        'target_id': targets,
        'Prot_1': [i / len(targets) for i in range(len(targets))],
        'Prot_2': [(i * 5 % 11) / 11 for i in range(len(targets))],
    })


class TestQsarPcmRealData(unittest.TestCase):
    """One test class per fixture would repeat this same body three times -
    parameterised via subTest instead.
    """

    mol_descs = None
    prot_descs = None

    def _run(self, func, data, **kwargs):
        with (
            patch(
                'src.papyrus_scripts.modelling.read_molecular_descriptors',
                return_value=self.mol_descs,
            ),
            patch(
                'src.papyrus_scripts.modelling.read_protein_descriptors',
                return_value=self.prot_descs,
            ),
        ):
            return func(data, verbose=False, **kwargs)

    def test_qsar_regressor_pandas_polars_parity(self):
        for fixture in FIXTURES:
            with self.subTest(fixture=fixture):
                df = _load_fixture(fixture)
                self.mol_descs = _synthetic_mol_descriptors(df)
                model_args = dict(model=DecisionTreeRegressor(random_state=0))
                results_pl, _ = self._run(qsar, df, **model_args)
                results_pd, _ = self._run(qsar, df.to_pandas(), **model_args)
                self.assertGreater(len(results_pl), 0)
                pd.testing.assert_frame_equal(
                    results_pl.reset_index(drop=True), results_pd.reset_index(drop=True),
                )
                self.assertGreater(len(results_pl.index.get_level_values('target').unique()), 1)

    def test_pcm_regressor_pandas_polars_parity(self):
        for fixture in FIXTURES:
            with self.subTest(fixture=fixture):
                df = _load_fixture(fixture)
                self.mol_descs = _synthetic_mol_descriptors(df)
                self.prot_descs = _synthetic_prot_descriptors(df)
                model_args = dict(model=DecisionTreeRegressor(random_state=0))
                results_pl, _ = self._run(pcm, df, **model_args)
                results_pd, _ = self._run(pcm, df.to_pandas(), **model_args)
                self.assertGreater(len(results_pl), 0)
                pd.testing.assert_frame_equal(
                    results_pl.reset_index(drop=True), results_pd.reset_index(drop=True),
                )

    def test_qsar_classifier_pandas_polars_parity(self):
        for fixture in FIXTURES:
            with self.subTest(fixture=fixture):
                df = _load_fixture(fixture)
                self.mol_descs = _synthetic_mol_descriptors(df)
                model_args = dict(model=DecisionTreeClassifier(random_state=0))
                results_pl, _ = self._run(qsar, df, **model_args)
                results_pd, _ = self._run(qsar, df.to_pandas(), **model_args)
                self.assertGreater(len(results_pl), 0)
                pd.testing.assert_frame_equal(
                    results_pl.reset_index(drop=True), results_pd.reset_index(drop=True),
                )
                # Multiple real targets must actually reach a fit, not just
                # hit the "insufficient data"/threshold-skip early returns.
                self.assertGreater(len(results_pl.index.get_level_values('target').unique()), 1)

    def test_pcm_classifier_pandas_polars_parity(self):
        for fixture in FIXTURES:
            with self.subTest(fixture=fixture):
                df = _load_fixture(fixture)
                self.mol_descs = _synthetic_mol_descriptors(df)
                self.prot_descs = _synthetic_prot_descriptors(df)
                model_args = dict(model=DecisionTreeClassifier(random_state=0))
                results_pl, _ = self._run(pcm, df, **model_args)
                results_pd, _ = self._run(pcm, df.to_pandas(), **model_args)
                self.assertGreater(len(results_pl), 0)
                pd.testing.assert_frame_equal(
                    results_pl.reset_index(drop=True), results_pd.reset_index(drop=True),
                )


if __name__ == '__main__':
    unittest.main()
