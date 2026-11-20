"""ctypes wrapper of spire-zero's Rust combat engine (libsts2ai_engine.so): import a live combat the host exported, clone it
as often as you like, play it out. A world is an opaque handle; free it (World.close, or use it as a context manager)."""
from __future__ import annotations

import ctypes
import json
import os

_SZ = os.environ.get("SPIRE_ZERO", os.path.expanduser("~/Programming/spire-zero"))
_LIB_PATH = os.environ.get("STS2AI_ENGINE_LIB", os.path.join(_SZ, "engine/target/release/libsts2ai_engine.so"))


class _Buf(ctypes.Structure):
    _fields_ = [("ptr", ctypes.POINTER(ctypes.c_uint8)), ("len", ctypes.c_size_t)]


_lib = None


def _load():
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(_LIB_PATH)
        u64, i32, c_sz, cp = ctypes.c_uint64, ctypes.c_int32, ctypes.c_size_t, ctypes.c_char_p
        lib.sz_world_import.restype, lib.sz_world_import.argtypes = u64, [cp, c_sz, i32]
        lib.sz_world_clone.restype, lib.sz_world_clone.argtypes = u64, [u64]
        lib.sz_world_free.argtypes = [u64]
        lib.sz_world_apply.restype, lib.sz_world_apply.argtypes = i32, [u64, cp, c_sz]
        lib.sz_world_status.restype, lib.sz_world_status.argtypes = i32, [u64]
        lib.sz_world_legal.restype, lib.sz_world_legal.argtypes = i32, [u64, ctypes.POINTER(_Buf)]
        lib.sz_world_observe.restype, lib.sz_world_observe.argtypes = i32, [u64, ctypes.POINTER(_Buf)]
        lib.sz_world_hidden_key.restype, lib.sz_world_hidden_key.argtypes = u64, [u64]
        lib.sz_buf_free.argtypes = [_Buf]
        lib.sz_last_error.restype, lib.sz_last_error.argtypes = i32, [ctypes.POINTER(_Buf)]
        _lib = lib
    return _lib


def _take(buf: _Buf) -> str:
    text = ctypes.string_at(buf.ptr, buf.len).decode() if buf.len else ""
    _load().sz_buf_free(buf)
    return text


class EngineError(RuntimeError):
    pass


class World:
    """A combat the engine holds. status: 0 in progress, 1 won, 2 lost."""
    __slots__ = ("h",)

    def __init__(self, handle: int):
        self.h = handle

    @classmethod
    def from_export(cls, export: bytes, sample: bool = False) -> "World":
        """`export` is the host's EngineExport (the decision's base64 `export`, decoded). sample=False keeps the exact hidden
        state (draw order and random streams): every clone draws the same cards, and branches differ only by the actions."""
        h = _load().sz_world_import(export, len(export), 1 if sample else 0)
        if not h:
            raise EngineError(f"import failed: {_last_error()}")
        return cls(h)

    def clone(self) -> "World":
        h = _load().sz_world_clone(self.h)
        if not h:
            raise EngineError(f"clone failed: {_last_error()}")
        return World(h)

    def status(self) -> int:
        return _load().sz_world_status(self.h)

    def legal(self) -> list[str]:
        b = _Buf()
        _load().sz_world_legal(self.h, ctypes.byref(b))
        return [x for x in _take(b).split("\n") if x]

    def observe(self) -> dict:
        b = _Buf()
        _load().sz_world_observe(self.h, ctypes.byref(b))
        return json.loads(_take(b))

    def apply(self, action: str) -> None:
        raw = action.encode()
        if _load().sz_world_apply(self.h, raw, len(raw)) != 0:
            raise EngineError(f"{action!r} refused: {_last_error()}")

    def hidden_key(self) -> int:
        return _load().sz_world_hidden_key(self.h)

    def close(self) -> None:
        if self.h:
            _load().sz_world_free(self.h)
            self.h = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _last_error() -> str:
    b = _Buf()
    _load().sz_last_error(ctypes.byref(b))
    return _take(b)
