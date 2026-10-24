"""Compatibility tests for upstream and imported-runtime checkpoint identities."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import os
from runtime.absa.paths import checkpoint_path
from scripts.update_kb import model_revision


class CheckpointPathTests(unittest.TestCase):
    def test_versioned_checkpoint_is_preferred(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'ABSA_VERSION':'v2'}):
            root = Path(folder)
            versioned = root / 'models/absa/v2/logreg/logreg.pkl'
            flat = root / 'models/logreg/logreg.pkl'
            for path in (versioned, flat):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture')
            self.assertEqual(checkpoint_path(root, 'logreg', 'logreg.pkl'), versioned)
            self.assertEqual(Path(model_revision(root, 'logreg')['path']), Path('absa/v2/logreg/logreg.pkl'))

    def test_flat_checkpoint_fallback(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'ABSA_VERSION':'v2'}):
            root = Path(folder)
            flat = root / 'models/phobert/phobert.pt'
            flat.parent.mkdir(parents=True)
            flat.write_bytes(b'fixture')
            self.assertEqual(checkpoint_path(root, 'phobert', 'phobert.pt'), flat)

    def test_version_override_and_invalid_version(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'ABSA_VERSION':'v1'}):
            path = checkpoint_path(Path(folder), 'logreg', 'logreg.pkl')
            self.assertIn('v1', path.parts)
        with patch.dict(os.environ, {'ABSA_VERSION':'../../outside'}):
            with self.assertRaises(ValueError):
                checkpoint_path(Path('.'), 'logreg', 'logreg.pkl')
