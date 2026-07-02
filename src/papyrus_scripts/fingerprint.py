# -*- coding: utf-8 -*-

from collections.abc import Callable
import json
import hashlib
from abc import ABC, abstractmethod

import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Avalon import pyAvalonTools
try:
    from openbabel import pybel
    HAS_PYBEL = True
except ImportError:
    HAS_PYBEL = False
try:
    from FPSim2.FPSim2lib.utils import BitStrToIntList, PyPopcount
    HAS_FPSIM2 = True
except ImportError:
    HAS_FPSIM2 = False


class Fingerprint(ABC):
    def __init__(self, name: str, params: dict, call_func: Callable):
        self.name = name
        self.params = params
        self.func = call_func
        # determine length
        self.length = None
        if "nBits" in params.keys():
            self.length = params["nBits"]
        elif "fpSize" in params.keys():
            self.length = params["fpSize"]
        elif self.name == "MACCSKeys":
            self.length = 166
        elif self.name == "FP2":
            self.length = 1024
        elif self.name == "FP3":
            self.length = 55
        elif self.name == "FP4":
            self.length = 307
        if not self.length:
            raise Exception("fingerprint size is not specified")
        self._hash = self.name + json.dumps(self.params, sort_keys=True)
        self._hash = hashlib.sha256((self._hash).encode()).digest()
        self._hash = np.frombuffer(self._hash, dtype=np.int64)
        self._hash = abs(np.sum(self._hash)) % 65537
        self._hash = f'{hex(self._hash)}'

    def __repr__(self):
        return f'{self.name}_{self.length}bits_{self._hash}'

    @classmethod
    def derived(cls):
        if not cls.__subclasses__():
            return cls
        subclasses = []
        for subclass in cls.__subclasses__():
            subclass_derived = subclass.derived()
            if isinstance(subclass_derived, list):
                subclasses.extend(subclass_derived)
            else:
                subclasses.append(subclass_derived)
        return subclasses

    @abstractmethod
    def get(self, mol: Chem.Mol) -> list[int]:
        """Get the bistring fingerprint of the molecule"""


class RDKitFingerprint(Fingerprint):
    def get(self, mol: Chem.Mol) -> list[int]:
        """Get the bistring fingerprint of the molecule and popcounts"""
        if not HAS_FPSIM2:
            raise ImportError('Some required dependencies are missing:\n\ttables, FPSim2')
        fp = BitStrToIntList(self.func(mol, **self.params).ToBitString())
        popcnt = PyPopcount(np.array(fp, dtype=np.uint64))
        return (*fp, popcnt)


class MACCSKeysFingerprint(RDKitFingerprint):
    def __init__(self):
        super().__init__('MACCSKeys', {}, rdMolDescriptors.GetMACCSKeysFingerprint)


class AvalonFingerprint(RDKitFingerprint):
    def __init__(self, nBits: int = 512, isQuery: bool = False, resetVect: bool = False, bitFlags: int = 15761407):
        super().__init__('Avalon',
                                                {'nBits': nBits,
                                                 'isQuery': isQuery,
                                                 'resetVect': resetVect,
                                                 'bitFlags': bitFlags},
                                                pyAvalonTools.GetAvalonFP)


class MorganFingerprint(RDKitFingerprint):
    def __init__(self, radius: int = 2, nBits: int = 2048, invariants: list = [], fromAtoms: list = [],
                 useChirality: bool = False, useBondTypes: bool = True, useFeatures: bool = False):
        super().__init__('Morgan',
                                                {'radius': radius,
                                                 'nBits': nBits,
                                                 'invariants': invariants,
                                                 'fromAtoms': fromAtoms,
                                                 'useChirality': useChirality,
                                                 'useBondTypes': useBondTypes,
                                                 'useFeatures': useFeatures},
                                                rdMolDescriptors.GetMorganFingerprintAsBitVect)


class TopologicalTorsionFingerprint(RDKitFingerprint):
    def __init__(self, nBits: int = 2048, targetSize: int = 4, fromAtoms: list = 0,
                 ignoreAtoms: list = 0, atomInvariants: list = 0, includeChirality: bool = False):
        super().__init__('TopologicalTorsion',
                         {"nBits": nBits,
                          "targetSize": targetSize,
                          "fromAtoms": fromAtoms,
                          "ignoreAtoms": ignoreAtoms,
                          "atomInvariants": atomInvariants,
                          "includeChirality": includeChirality, },
                         rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect)


class AtomPairFingerprint(RDKitFingerprint):
    def __init__(self, nBits: int = 2048, minLength: int = 1, maxLength: int = 30,
                 fromAtoms: list = 0, ignoreAtoms: list = 0, atomInvariants: list = 0,
                 nBitsPerEntry: int = 4, includeChirality: bool = False,
                 use2D: bool = True, confId: int = -1):
        super().__init__('AtomPair',
                                                  {"nBits": nBits,
                                                   "minLength": minLength,
                                                   "maxLength": maxLength,
                                                   "fromAtoms": fromAtoms,
                                                   "ignoreAtoms": ignoreAtoms,
                                                   "atomInvariants": atomInvariants,
                                                   "nBitsPerEntry": nBitsPerEntry,
                                                   "includeChirality": includeChirality,
                                                   "use2D": use2D,
                                                   "confId": confId},
                                                  rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect)


class RDKitTopologicalFingerprint(RDKitFingerprint):
    def __init__(self, fpSize: int = 2048, minPath: int = 1, maxPath: int = 7, nBitsPerHash: int = 2,
                 useHs: bool = True, tgtDensity: float = 0.0, minSize: int = 128,
                 branchedPaths: bool = True, useBondOrder: bool = True, atomInvariants: list = 0,
                 fromAtoms: list = 0, atomBits: list = None, bitInfo: list = None):
        super().__init__('RDKFingerprint',
                                                          {"minPath": minPath,
                                                           "maxPath": maxPath,
                                                           "fpSize": fpSize,
                                                           "nBitsPerHash": nBitsPerHash,
                                                           "useHs": useHs,
                                                           "tgtDensity": tgtDensity,
                                                           "minSize": minSize,
                                                           "branchedPaths": branchedPaths,
                                                           "useBondOrder": useBondOrder,
                                                           "atomInvariants": atomInvariants,
                                                           "fromAtoms": fromAtoms,
                                                           "atomBits": atomBits,
                                                           "bitInfo": bitInfo},
                                                          Chem.RDKFingerprint)


class RDKPatternFingerprint(RDKitFingerprint):
    def __init__(self, fpSize: int = 2048, atomCounts: list = [], setOnlyBits: list = None):
        super().__init__('RDKPatternFingerprint',
                                                    {'fpSize': fpSize,
                                                     'atomCounts': atomCounts,
                                                     'setOnlyBits': setOnlyBits},
                                                    Chem.PatternFingerprint)


class OBFingerprint(Fingerprint):
    def __init__(self, name: str, params: dict, call_func: Callable):
        if not HAS_PYBEL and not HAS_FPSIM2:
            raise ImportError('Some required dependencies are missing:\n\topenbabel, FPSim2')
        elif not HAS_PYBEL:
            raise ImportError('Some required dependencies are missing:\n\topenbabel')
        elif not HAS_FPSIM2:
            raise ImportError('Some required dependencies are missing:\n\tFPSim2')
        super().__init__(name, params, call_func)

    def get(self, mol: Chem.Mol) -> list[int]:
        """Get the bistring fingerprint of the molecule and popcounts"""
        binvec = DataStructs.ExplicitBitVect(self.length)
        obmol = pybel.readstring('smi', Chem.MolToSmiles(mol))
        binvec.SetBitsFromList([x - 1 for x in obmol.calcfp(self.func).bits])
        fp = BitStrToIntList(binvec.ToBitString())
        popcnt = PyPopcount(np.array(fp, dtype=np.uint64))
        return (*fp, popcnt)


class FP2Fingerprint(OBFingerprint):
    def __init__(self):
        super().__init__('FP2',
                                             {},
                                             'FP2')


class FP3Fingerprint(OBFingerprint):
    def __init__(self):
        super().__init__('FP3',
                                             {},
                                             'FP3')


class FP4Fingerprint(OBFingerprint):
    def __init__(self):
        super().__init__('FP4',
                                             {},
                                             'FP4')


def get_fp_from_name(fp_name, **kwargs):
    """Get the fingerprint TYPE corresponding to a name
    :param fp_name: Name of the fingerprint
    :param kwargs: parameters specific to the desired fingerprint
    :return: fingerprint instance
    """
    fps = {}
    for fp_cls in Fingerprint.derived():
        try:
            fps[fp_cls().name] = fp_cls
        except ImportError:
            # Skip fingerprints whose optional dependencies (openbabel, FPSim2)
            # are not installed instead of letting them break the lookup of
            # every other, unrelated fingerprint type.
            continue
    if fp_name not in fps.keys():
        raise ValueError(f'Fingerprint {fp_name!r} not available')
    return fps[fp_name](**kwargs)