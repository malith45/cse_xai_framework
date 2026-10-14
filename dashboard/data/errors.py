"""
Typed data-layer errors.

Pages never see raw driver exceptions. The resilient repository wrapper turns
connection failures into `DataUnavailable` after retrying, and repositories
raise `NoData` when a query is valid but simply has no rows (no prediction for
a weekend date, no news for a stock, risk engine not yet run, ...).

Why the classes live in a synthetic module: Streamlit's hot reload unloads
*every* module under the app directory whenever any file changes, but objects
held in `st.cache_resource` (the repository) keep references to the old
classes. An `except NoData` in freshly reloaded code would then miss a `NoData`
raised by the cached repository. Anchoring the classes in a module that has no
file keeps their identity stable across reloads. In production nothing is ever
reloaded, so this is purely a developer-experience safeguard.
"""
from __future__ import annotations

import sys
import types

_ANCHOR = "_cse_error_types"


def _define() -> types.ModuleType:
    class DataUnavailable(Exception):
        """The data service could not be reached after every retry."""

    class NoData(LookupError):
        """The request was valid but nothing is stored for that selection."""

    module = types.ModuleType(_ANCHOR)
    module.DataUnavailable = DataUnavailable  # type: ignore[attr-defined]
    module.NoData = NoData                    # type: ignore[attr-defined]
    for cls in (DataUnavailable, NoData):
        cls.__module__ = __name__
    sys.modules[_ANCHOR] = module
    return module


_types = sys.modules.get(_ANCHOR) or _define()
DataUnavailable = _types.DataUnavailable
NoData = _types.NoData

__all__ = ["DataUnavailable", "NoData"]
