# -*- coding: utf-8 -*-

"""Print ``if TYPE_CHECKING:`` stub declarations for oop.py's dynamically generated filter methods.

``PapyrusDataFilter``/``FPSubSim2Engine`` attach filter methods via
``setattr()`` (see ``_generate_filters`` in oop.py), which keeps them DRY
but invisible to IDE autocomplete/mypy. This re-derives their real
signatures and prints ``def ...: ...`` stubs to paste into each class's
``if TYPE_CHECKING:`` block.

Re-run and re-paste whenever a ``preprocess.py`` filter's signature changes.
Usage: ``python scripts/generate_oop_filter_stubs.py``
"""

import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from papyrus_scripts import oop  # noqa: E402

#: Strips any papyrus_scripts submodule path to the bare class name (e.g.
#: "src.papyrus_scripts.fingerprint.Fingerprint" -> "Fingerprint").
_QUALPATH_RE = re.compile(r'\b(?:\w+\.)*papyrus_scripts(?:\.\w+)*\.(\w+)\b')


def _render_annotation(annotation: object) -> str:
    if annotation is inspect.Parameter.empty:
        return ''
    # A plain class (not a Union/generic) - str() on it gives "<class 'str'>",
    # so use its bare name directly instead.
    if isinstance(annotation, type):
        return annotation.__name__
    text = str(annotation).replace('typing.', '')
    return _QUALPATH_RE.sub(r'\1', text)


def _render_default(default: object) -> str:
    if default is inspect.Parameter.empty:
        return ''
    return f' = {default!r}'


def _render_method(name: str, method: object) -> str:
    sig = inspect.signature(method)
    params = []
    for pname, p in sig.parameters.items():
        if pname == 'self':
            params.append('self')
            continue
        annotation = _render_annotation(p.annotation)
        piece = f'{pname}: {annotation}' if annotation else pname
        params.append(piece + _render_default(p.default))
    summary = (inspect.getdoc(method) or '').split('\n', 1)[0]
    return (
        f'    def {name}({", ".join(params)}) -> PapyrusDataset:\n'
        f'        """{summary}"""\n'
        f'        ...\n'
    )


def main() -> None:
    """Print stub declarations for every generated filter method, grouped by class."""
    for cls, specs in (
        (oop.PapyrusDataFilter, oop._PAPYRUS_DATA_FILTER_SPECS),
        (oop.FPSubSim2Engine, oop._FPSUBSIM2_SPECS),
    ):
        print(f'# --- paste into `if TYPE_CHECKING:` in {cls.__name__} ---')
        for spec in specs:
            name = spec.name or spec.target_name
            print(_render_method(name, getattr(cls, name)))


if __name__ == '__main__':
    main()
