from __future__ import annotations
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = ROOT / "src"
DATA_CONFIG_PATH = ROOT / "config" / "data.json"
RAW_DATA_DIR = ROOT / "data" / "raw"
INPUT_CACHE_DIR = ROOT / "data" / "cache"
RUN_DIR = ROOT / "results" / "__unconfigured__"
CPU_THREADS = 6
_CONFIGURED = False


def configure(run_id, data_dir=None, threads=6):

    global RAW_DATA_DIR, RUN_DIR, CPU_THREADS, _CONFIGURED
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("run-id must be 1-80 ASCII letters/digits/underscore/hyphen; start with a letter or digit")
    if run_id.upper() in {"CON", "PRN", "AUX", "NUL", *("COM"+str(i) for i in range(1,10)), *("LPT"+str(i) for i in range(1,10))}:
        raise ValueError("Reserved operating-system filename")
    if _CONFIGURED and (RUN_DIR.name != run_id or CPU_THREADS != int(threads) or (data_dir and RAW_DATA_DIR != Path(data_dir).resolve())):
        raise RuntimeError("Configure once per process; start a new process for a different run")
    CPU_THREADS = int(threads)
    if CPU_THREADS < 1 or CPU_THREADS > 128:
        raise ValueError("threads must be between 1 and 128")
    RAW_DATA_DIR = Path(data_dir).resolve() if data_dir else ROOT / "data" / "raw"
    RUN_DIR = ROOT / "results" / run_id
    os.environ["DENGUE_CPU_THREADS"] = str(CPU_THREADS)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    _CONFIGURED = True
    return RUN_DIR


def require_configured():
    if not _CONFIGURED:
        raise RuntimeError("Configure a named run through run.py before performing an experiment")
