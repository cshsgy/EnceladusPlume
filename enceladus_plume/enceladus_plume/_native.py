"""Optional C++ performance core loader.

Attempts to import the compiled ``_enceladus_core`` extension (built from
``cpp/``). If it is unavailable, ``CORE`` is ``None`` and the pure-Python
implementations are used. Nothing in the package should *require* the
extension — it is an accelerator only.

Build it with::

    cmake -S cpp -B cpp/build -DPython_EXECUTABLE=$(which python3)
    cmake --build cpp/build

which places the extension next to this package so the import below succeeds.
"""

from __future__ import annotations

try:  # pragma: no cover - exercised only when the extension is built
    from . import _enceladus_core as CORE  # type: ignore
    HAVE_NATIVE = True
except ImportError:  # pragma: no cover
    CORE = None  # type: ignore
    HAVE_NATIVE = False

__all__ = ["CORE", "HAVE_NATIVE"]
