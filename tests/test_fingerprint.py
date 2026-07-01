# -*- coding: utf-8 -*-

"""Unit tests for papyrus_scripts.fingerprint.

Fingerprint *construction* and hashing only depend on RDKit (always
available). Actually computing bitstrings via .get() additionally requires
the optional FPSim2 dependency, and OBFingerprint-based fingerprints (FP2,
FP3, FP4) additionally require openbabel; those tests are skipped when the
optional dependency is not installed rather than being mocked, since the
missing-dependency behaviour itself is part of what's under test.
"""

import unittest

from rdkit import Chem

from src.papyrus_scripts import fingerprint as fp

FPSIM2_AVAILABLE = fp.HAS_FPSIM2
PYBEL_AVAILABLE = fp.HAS_PYBEL


class TestFingerprintConstruction(unittest.TestCase):

    def test_length_from_nbits(self):
        f = fp.MorganFingerprint(nBits=512)
        self.assertEqual(f.length, 512)

    def test_length_from_fpsize(self):
        f = fp.RDKitTopologicalFingerprint(fpSize=1024)
        self.assertEqual(f.length, 1024)

    def test_length_hardcoded_for_maccs(self):
        f = fp.MACCSKeysFingerprint()
        self.assertEqual(f.length, 166)

    def test_repr_contains_name_and_length_and_hash(self):
        f = fp.MorganFingerprint(nBits=256)
        r = repr(f)
        self.assertIn('Morgan', r)
        self.assertIn('256bits', r)
        self.assertIn(f._hash, r)

    def test_hash_is_deterministic_for_same_params(self):
        f1 = fp.MorganFingerprint(radius=2, nBits=2048)
        f2 = fp.MorganFingerprint(radius=2, nBits=2048)
        self.assertEqual(f1._hash, f2._hash)
        self.assertEqual(repr(f1), repr(f2))

    def test_hash_differs_for_different_params(self):
        f1 = fp.MorganFingerprint(radius=2, nBits=2048)
        f2 = fp.MorganFingerprint(radius=3, nBits=2048)
        self.assertNotEqual(f1._hash, f2._hash)

    def test_hash_differs_across_fingerprint_types(self):
        f1 = fp.MorganFingerprint()
        f2 = fp.AvalonFingerprint()
        self.assertNotEqual(f1._hash, f2._hash)


class TestFingerprintDerived(unittest.TestCase):

    def test_derived_lists_all_leaf_subclasses(self):
        names = {cls.__name__ for cls in fp.Fingerprint.derived()}
        expected = {
            'MorganFingerprint', 'AvalonFingerprint', 'MACCSKeysFingerprint',
            'TopologicalTorsionFingerprint', 'AtomPairFingerprint',
            'RDKitTopologicalFingerprint', 'RDKPatternFingerprint',
            'FP2Fingerprint', 'FP3Fingerprint', 'FP4Fingerprint',
        }
        self.assertEqual(names, expected)

    def test_derived_on_leaf_class_returns_itself(self):
        self.assertEqual(fp.MorganFingerprint.derived(), fp.MorganFingerprint)


class TestGetFpFromName(unittest.TestCase):

    def test_get_available_fingerprint_by_name(self):
        f = fp.get_fp_from_name('Morgan')
        self.assertIsInstance(f, fp.MorganFingerprint)

    def test_get_available_fingerprint_forwards_kwargs(self):
        f = fp.get_fp_from_name('Morgan', nBits=128)
        self.assertEqual(f.length, 128)

    def test_unknown_name_raises_value_error(self):
        with self.assertRaises(ValueError):
            fp.get_fp_from_name('NotAFingerprint')

    @unittest.skipIf(FPSIM2_AVAILABLE and PYBEL_AVAILABLE,
                     'requires openbabel and/or FPSim2 to be missing')
    def test_unavailable_fingerprint_raises_value_error_not_import_error(self):
        # Regression test: get_fp_from_name used to instantiate every known
        # Fingerprint subclass to build its name lookup, so requesting *any*
        # fingerprint (even one with no optional dependencies) would crash
        # with an unrelated ImportError if openbabel/FPSim2 were missing.
        with self.assertRaises(ValueError):
            fp.get_fp_from_name('FP2')
        # And fingerprints with no optional runtime dependency for construction
        # must still be reachable.
        f = fp.get_fp_from_name('MACCSKeys')
        self.assertIsInstance(f, fp.MACCSKeysFingerprint)


@unittest.skipUnless(FPSIM2_AVAILABLE, 'requires FPSim2')
class TestFingerprintGetWithFPSim2(unittest.TestCase):

    def test_get_returns_bits_and_popcount(self):
        mol = Chem.MolFromSmiles('CCO')
        f = fp.MorganFingerprint(radius=2, nBits=64)
        result = f.get(mol)
        # bits + a trailing popcount value.
        self.assertEqual(len(result), 3)  # 64 bits packed into 2x32-bit ints + popcount


@unittest.skipIf(FPSIM2_AVAILABLE, 'documents behaviour when FPSim2 is missing')
class TestFingerprintGetWithoutFPSim2(unittest.TestCase):

    def test_get_raises_import_error(self):
        mol = Chem.MolFromSmiles('CCO')
        f = fp.MorganFingerprint(nBits=64)
        with self.assertRaises(ImportError):
            f.get(mol)


if __name__ == '__main__':
    unittest.main()