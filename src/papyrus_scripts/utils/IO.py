# -*- coding: utf-8 -*-

"""IO functions."""

import hashlib
import importlib
import inspect
import json
import shutil


def sha256sum(filename, blocksize=None):
    if blocksize is None:
        blocksize = 65536
    hash = hashlib.sha256()
    with open(filename, "rb") as fh:
        for block in iter(lambda: fh.read(blocksize), b""):
            hash.update(block)
    return hash.hexdigest()


def assert_sha256sum(filename, sha256, blocksize=None):
    if not (isinstance(sha256, str) and len(sha256) == 64):
        raise ValueError("SHA256 must be 64 chars: {}".format(sha256))
    sha256_actual = sha256sum(filename, blocksize)
    return sha256_actual == sha256


def write_jsonfile(data: object, json_outfile: str) -> None:
    """Write a json object to a file with lazy formatting."""
    with open(json_outfile, 'w') as outfile:
        json.dump(data, outfile, indent=4)


def read_jsonfile(json_infile: str) -> object:
    """Read in a json file and return the json object."""
    with open(json_infile) as infile:
        data = json.load(infile)
    return data


class TypeEncoder(json.JSONEncoder):
    """Custom json encoder to support types as values."""

    def default(self, obj):
        """Add support if value if a type."""
        if isinstance(obj, type):
            return {'__type__': {'module': inspect.getmodule(obj).__name__,
                                 'type': obj.__name__}
                    }
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


class TypeDecoder(json.JSONDecoder):
    """Custom json decoder to support types as values."""

    def __init__(self, *args, **kwargs):
        """Simple json decoder handling types as values."""
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        """Handle types."""
        if '__type__' not in obj:
            return obj
        module = obj['__type__']['module']
        type_ = obj['__type__']['type']
        if module == 'builtins':
            return getattr(__builtins__, type_)
        loaded_module = importlib.import_module(module)
        return getattr(loaded_module, type_)



def enough_disk_space(destination: str,
                      required: int,
                      margin: float = 0.10):
    """Check disk has enough space.
    
    :param destination: folder to check
    :param required: space required in bytes
    :param margin: percent of free disk space once file is written
    """
    total, _, free = shutil.disk_usage(destination)
    return free - required > margin * total