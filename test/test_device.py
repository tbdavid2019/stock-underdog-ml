"""
Unit tests for DeviceManager
"""
import unittest
import os
from unittest.mock import patch, MagicMock
from core.device import DeviceManager
import core.device as device_module


class TestDeviceManager(unittest.TestCase):

    def test_explicit_cpu(self):
        dev = DeviceManager.get_torch_device(preferred="cpu")
        self.assertEqual(getattr(dev, "type", str(dev)), "cpu")

    def test_auto_detection(self):
        dev = DeviceManager.get_torch_device(preferred="auto")
        self.assertIn(getattr(dev, "type", str(dev)), ["cuda", "mps", "cpu"])

    def test_cuda_fallback_when_unavailable(self):
        with patch.object(device_module, "HAS_TORCH", True), \
             patch.object(device_module, "torch", create=True) as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.device.side_effect = lambda x: MagicMock(type=x)
            dev = DeviceManager.get_torch_device(preferred="cuda")
            self.assertEqual(getattr(dev, "type", str(dev)), "cpu")

    def test_cuda_success_when_available(self):
        with patch.object(device_module, "HAS_TORCH", True), \
             patch.object(device_module, "torch", create=True) as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.device.side_effect = lambda x: MagicMock(type=x)
            dev = DeviceManager.get_torch_device(preferred="cuda")
            self.assertEqual(getattr(dev, "type", str(dev)), "cuda")

    def test_mps_success_when_available(self):
        with patch.object(device_module, "HAS_TORCH", True), \
             patch.object(device_module, "torch", create=True) as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = True
            mock_torch.device.side_effect = lambda x: MagicMock(type=x)
            dev = DeviceManager.get_torch_device(preferred="mps")
            self.assertEqual(getattr(dev, "type", str(dev)), "mps")

    def test_device_info(self):
        info = DeviceManager.get_device_info()
        self.assertIn("device", info)
        self.assertIn("type", info)
        self.assertIn("name", info)


if __name__ == "__main__":
    unittest.main()
