import os
from pathlib import Path


def env_key(name: str) -> str | None:
    """An API key from the environment, else from .env at the repo root (gitignored)."""
    if os.environ.get(name):
        return os.environ[name]
    env = Path(__file__).resolve().parents[3] / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        k, _, v = line.partition("=")
        if k.strip() == name:
            return v.strip().strip("'\"") or None
    return None
