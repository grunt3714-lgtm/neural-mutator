from __future__ import annotations

import pickle
from typing import Any


def dumps(obj: Any) -> bytes:
    # protocol 5 supports out-of-band buffers, but plain bytes is fine for now.
    return pickle.dumps(obj, protocol=5)


def loads(b: bytes) -> Any:
    return pickle.loads(b)
