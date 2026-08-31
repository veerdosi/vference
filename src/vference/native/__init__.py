from __future__ import annotations


def extension():
    try:
        from . import _vference_native
    except ImportError as error:
        raise RuntimeError(
            "the native stable-slot extension is not built; run `make -C native build`"
        ) from error
    return _vference_native
