"""Resolve versioned upstream checkpoints and flat Tiki_ABSA checkpoints."""
import os
from pathlib import Path


def checkpoint_path(root, model, filename, version=None):
    version = os.getenv('ABSA_VERSION', version or 'v2')
    if version not in ('v1', 'v2'):
        raise ValueError('ABSA_VERSION must be v1 or v2')
    base = Path(root) / 'models'
    candidates = [base / 'absa' / version / model / filename, base / model / filename]
    return next((path for path in candidates if path.exists()), candidates[0])
