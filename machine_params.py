"""Confidential parameters of the test excavator, stored encrypted in machine_params.enc.

The excavator's kinematic geometry, dynamic parameters, constraint limits and the trained
digging-resistance and bucket-fill-ratio models belong to the industrial partner and are not
published in plain text. They are needed only to evaluate trajectories (objective and constraint
checks, PSO, the field experiment); training and the stored results do not need them.

Provide the key in one of these ways (the key is available from the corresponding author on
reasonable request):
  * environment variable PGCD_PARAMS_KEY holding the key,
  * environment variable PGCD_PARAMS_KEY_FILE pointing to a file holding the key,
  * the file ~/.pgcd/machine_params.key.
"""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

BUNDLE = Path(__file__).resolve().with_name("machine_params.enc")
KEY_FILE = Path.home() / ".pgcd" / "machine_params.key"
_DATA = None


def _key() -> bytes:
    if os.environ.get("PGCD_PARAMS_KEY"):
        return os.environ["PGCD_PARAMS_KEY"].strip().encode()
    path = Path(os.environ.get("PGCD_PARAMS_KEY_FILE", KEY_FILE))
    if not path.exists():
        raise RuntimeError(
            "The confidential machine parameters are encrypted (machine_params.enc) and no key was found. "
            "Set PGCD_PARAMS_KEY or PGCD_PARAMS_KEY_FILE, or place the key in ~/.pgcd/machine_params.key. "
            "The key is available from the corresponding author on reasonable request.")
    return path.read_bytes().strip()


def _data() -> dict:
    global _DATA
    if _DATA is None:
        from cryptography.fernet import Fernet, InvalidToken

        try:
            _DATA = json.loads(Fernet(_key()).decrypt(BUNDLE.read_bytes()))
        except InvalidToken:
            raise RuntimeError("machine_params.enc could not be decrypted with the given key") from None
    return _DATA


def get(section: str) -> dict:
    """Parameter section: 'constraints', 'normalization' or 'bucket'."""
    return _data()["params"][section]


def model(name: str) -> io.BytesIO:
    """Serialized model weights ('model_F' resistance, 'model_v' fill ratio) for torch.load."""
    return io.BytesIO(base64.b64decode(_data()["models"][name]))


def exec_module(path: str, namespace: dict) -> None:
    """Run an encrypted module source (e.g. 'PhysicsModel/Kinematics.py') in the namespace of its stub."""
    exec(compile(_data()["modules"][path], f"<machine_params:{path}>", "exec"), namespace)
