# -*- coding: utf-8 -*-

"""Fingerprint classes wrapping RDKit/Avalon/OpenBabel fingerprint algorithms."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import rdMolDescriptors

try:
    from openbabel import pybel
    HAS_PYBEL = True
except ImportError:  # pragma: no cover - only taken when openbabel isn't installed
    HAS_PYBEL = False
try:
    from FPSim2.FPSim2lib.utils import BitStrToIntList, PyPopcount
    HAS_FPSIM2 = True  # pragma: no cover - only taken when FPSim2 is installed
except ImportError:
    HAS_FPSIM2 = False


class Fingerprint(ABC):
    """Base class for a molecular fingerprint algorithm and its parameters."""

    def __init__(self, name: str, params: dict, call_func: Callable | str) -> None:
        """Store *name*/*params*/*call_func* and derive the bit length and a parameter hash."""
        self.name = name
        self.params = params
        self.func = call_func
        # determine length
        length: int | None = None
        if "nBits" in params.keys():
            length = params["nBits"]
        elif "fpSize" in params.keys():
            length = params["fpSize"]
        elif self.name == "MACCSKeys":
            length = 166
        elif self.name == "FP2":
            length = 1024
        elif self.name == "FP3":
            length = 55
        elif self.name == "FP4":
            length = 307
        if not length:
            raise ValueError("fingerprint size is not specified")
        self.length: int = length
        digest = hashlib.sha256((self.name + json.dumps(self.params, sort_keys=True)).encode()).digest()
        hash_ints = np.frombuffer(digest, dtype=np.int64)
        hash_int = abs(int(np.sum(hash_ints))) % 65537
        self._hash = f'{hex(hash_int)}'

    def __repr__(self) -> str:
        """Return "<name>_<length>bits_<hash>"."""
        return f'{self.name}_{self.length}bits_{self._hash}'

    @classmethod
    def derived(cls) -> type[Fingerprint] | list[type[Fingerprint]]:
        """Return every leaf (non-abstract) subclass, recursively.

        A leaf class (no subclasses of its own) returns itself directly
        rather than a single-element list - callers that only ever invoke
        this on :class:`Fingerprint` itself (which always has subclasses)
        get a flat ``list``; the bare-class case only surfaces when calling
        ``derived()`` directly on a leaf class.
        """
        if not cls.__subclasses__():
            return cls
        subclasses: list[type[Fingerprint]] = []
        for subclass in cls.__subclasses__():
            subclass_derived = subclass.derived()
            if isinstance(subclass_derived, list):
                subclasses.extend(subclass_derived)
            else:
                subclasses.append(subclass_derived)
        return subclasses

    @abstractmethod
    def get(self, mol: Chem.Mol) -> list[int]:
        """Get the bitstring fingerprint of the molecule."""


class RDKitFingerprint(Fingerprint):
    """Fingerprint computed via an RDKit callable."""

    def get(self, mol: Chem.Mol) -> list[int]:
        """Get the bitstring fingerprint of the molecule and popcount."""
        if not HAS_FPSIM2:
            raise ImportError('Some required dependencies are missing:\n\ttables, FPSim2')
        # RDKitFingerprint always constructs itself with a real callable (see
        # each subclass' __init__) - only OBFingerprint uses the str variant.
        # Lines below require FPSim2 installed (see the raise above).
        if not callable(self.func):  # pragma: no cover
            raise TypeError('self.func must be callable')
        fp = BitStrToIntList(self.func(mol, **self.params).ToBitString())  # pragma: no cover
        popcnt = PyPopcount(np.array(fp, dtype=np.uint64))  # pragma: no cover
        return [*fp, popcnt]  # pragma: no cover


class MACCSKeysFingerprint(RDKitFingerprint):
    """166-bit MACCS structural keys fingerprint."""

    def __init__(self) -> None:
        """Configure the (parameter-free) MACCS keys fingerprint."""
        super().__init__('MACCSKeys', {}, rdMolDescriptors.GetMACCSKeysFingerprint)


class AvalonFingerprint(RDKitFingerprint):
    """Avalon fingerprint (substructure/similarity fingerprint from the Avalon toolkit)."""

    def __init__(self, nBits: int = 512, isQuery: bool = False, resetVect: bool = False,
                 bitFlags: int = 15761407) -> None:
        """Configure the Avalon fingerprint's bit vector size and flags."""
        super().__init__('Avalon',
                                                {'nBits': nBits,
                                                 'isQuery': isQuery,
                                                 'resetVect': resetVect,
                                                 'bitFlags': bitFlags},
                                                pyAvalonTools.GetAvalonFP)


class MorganFingerprint(RDKitFingerprint):
    """Morgan (circular, ECFP-like) fingerprint."""

    def __init__(self, radius: int = 2, nBits: int = 2048, invariants: list | None = None,
                 fromAtoms: list | None = None, useChirality: bool = False, useBondTypes: bool = True,
                 useFeatures: bool = False) -> None:
        """Configure the Morgan fingerprint's radius, size and invariants."""
        invariants = invariants if invariants is not None else []
        fromAtoms = fromAtoms if fromAtoms is not None else []
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
    """Topological-torsion fingerprint."""

    def __init__(self, nBits: int = 2048, targetSize: int = 4, fromAtoms: list | int = 0,
                 ignoreAtoms: list | int = 0, atomInvariants: list | int = 0,
                 includeChirality: bool = False) -> None:
        """Configure the topological-torsion fingerprint."""
        super().__init__('TopologicalTorsion',
                         {"nBits": nBits,
                          "targetSize": targetSize,
                          "fromAtoms": fromAtoms,
                          "ignoreAtoms": ignoreAtoms,
                          "atomInvariants": atomInvariants,
                          "includeChirality": includeChirality },
                         rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect)


class AtomPairFingerprint(RDKitFingerprint):
    """Atom-pair fingerprint."""

    def __init__(self, nBits: int = 2048, minLength: int = 1, maxLength: int = 30,
                 fromAtoms: list | int = 0, ignoreAtoms: list | int = 0, atomInvariants: list | int = 0,
                 nBitsPerEntry: int = 4, includeChirality: bool = False,
                 use2D: bool = True, confId: int = -1) -> None:
        """Configure the atom-pair fingerprint."""
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
    """RDKit's own (Daylight-like) topological fingerprint."""

    def __init__(self, fpSize: int = 2048, minPath: int = 1, maxPath: int = 7, nBitsPerHash: int = 2,
                 useHs: bool = True, tgtDensity: float = 0.0, minSize: int = 128,
                 branchedPaths: bool = True, useBondOrder: bool = True, atomInvariants: list | int = 0,
                 fromAtoms: list | int = 0, atomBits: list | None = None, bitInfo: list | None = None) -> None:
        """Configure RDKit's topological fingerprint."""
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
    """Pattern fingerprint used by RDKit for substructure screening."""

    def __init__(self, fpSize: int = 2048, atomCounts: list | None = None,
                 setOnlyBits: list | None = None) -> None:
        """Configure the pattern fingerprint."""
        atomCounts = atomCounts if atomCounts is not None else []
        super().__init__('RDKPatternFingerprint',
                                                    {'fpSize': fpSize,
                                                     'atomCounts': atomCounts,
                                                     'setOnlyBits': setOnlyBits},
                                                    Chem.PatternFingerprint)


class OBFingerprint(Fingerprint):
    """Fingerprint computed via OpenBabel, requiring both openbabel and FPSim2."""

    def __init__(self, name: str, params: dict, call_func: str) -> None:
        """Validate openbabel/FPSim2 are installed, then configure like the base class."""
        if not HAS_PYBEL and not HAS_FPSIM2:
            raise ImportError('Some required dependencies are missing:\n\topenbabel, FPSim2')  # pragma: no cover
        elif not HAS_PYBEL:
            raise ImportError('Some required dependencies are missing:\n\topenbabel')  # pragma: no cover
        elif not HAS_FPSIM2:
            raise ImportError('Some required dependencies are missing:\n\tFPSim2')
        super().__init__(name, params, call_func)  # pragma: no cover - needs both deps installed

    def get(self, mol: Chem.Mol) -> list[int]:
        """Get the bitstring fingerprint of the molecule and popcount."""
        binvec = DataStructs.ExplicitBitVect(self.length)  # pragma: no cover
        obmol = pybel.readstring('smi', Chem.MolToSmiles(mol))  # pragma: no cover
        binvec.SetBitsFromList([x - 1 for x in obmol.calcfp(self.func).bits])  # pragma: no cover
        fp = BitStrToIntList(binvec.ToBitString())  # pragma: no cover
        popcnt = PyPopcount(np.array(fp, dtype=np.uint64))  # pragma: no cover
        return [*fp, popcnt]  # pragma: no cover


class FP2Fingerprint(OBFingerprint):
    """OpenBabel FP2 fingerprint: 1024-bit, path-based (Daylight-like)."""

    def __init__(self) -> None:
        """Configure the (parameter-free) FP2 fingerprint."""
        super().__init__('FP2',
                                             {},
                                             'FP2')


class FP3Fingerprint(OBFingerprint):
    """OpenBabel FP3 fingerprint: 55-bit, based on a small set of SMARTS patterns."""

    def __init__(self) -> None:
        """Configure the (parameter-free) FP3 fingerprint."""
        super().__init__('FP3',
                                             {},
                                             'FP3')


class FP4Fingerprint(OBFingerprint):
    """OpenBabel FP4 fingerprint: 307-bit, based on a larger SMARTS pattern set than FP3."""

    def __init__(self) -> None:
        """Configure the (parameter-free) FP4 fingerprint."""
        super().__init__('FP4',
                                             {},
                                             'FP4')


def get_fp_from_name(fp_name: str, **kwargs: Any) -> Fingerprint:
    """Get a Fingerprint instance for the given name.

    :param fp_name: name of the fingerprint (e.g. 'Morgan', 'MACCSKeys')
    :param kwargs: parameters specific to the desired fingerprint
    :return: fingerprint instance
    """
    fps: dict[str, type[Fingerprint]] = {}
    # Fingerprint (unlike a leaf subclass) always has subclasses, so
    # derived() always returns a list here.
    for fp_cls in Fingerprint.derived():  # type: ignore[union-attr]
        try:
            # Every concrete subclass overrides __init__ with its own
            # no-argument signature; mypy only sees the abstract base's.
            fps[fp_cls().name] = fp_cls  # type: ignore[call-arg]
        except ImportError:
            # Skip fingerprints whose optional dependencies (openbabel, FPSim2)
            # are not installed instead of letting them break the lookup of
            # every other, unrelated fingerprint type.
            continue
    if fp_name not in fps.keys():
        raise ValueError(f'Fingerprint {fp_name!r} not available')
    return fps[fp_name](**kwargs)
