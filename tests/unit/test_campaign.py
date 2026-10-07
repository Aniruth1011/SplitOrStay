import tempfile
import unittest
from pathlib import Path

from heterokv.artifacts import sha256
from heterokv.campaign import sweep, verify_frozen


class CampaignTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_test_set_sweep(self):
        with self.assertRaises(ValueError):
            await sweep({"stage": "test"})
        with self.assertRaises(ValueError):
            await sweep({"stage": "validation", "dataset_id": "azure2024"})

    def test_frozen_input_and_seed_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("backend: mock\n")
            manifest = {"frozen": True, "config": str(path),
                        "input_hashes": {str(path): sha256(path)},
                        "validation_seeds": [101], "test_seeds": [201, 202, 203, 204, 205]}
            self.assertEqual(verify_frozen(manifest)["backend"], "mock")
            with self.assertRaises(ValueError):
                verify_frozen(manifest | {"test_seeds": [101, 202, 203, 204, 205]})
            path.write_text("backend: changed\n")
            with self.assertRaises(ValueError):
                verify_frozen(manifest)
