"""
Device Manager for Hardware Acceleration (CUDA / Apple Silicon MPS / CPU).
Provides unified runtime detection, manual device selection, and graceful fallback.
"""
import os
import logging
from typing import Optional, Any

logger = logging.getLogger("stock_app.device")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class DeviceManager:
    """Unified Hardware Device Manager for PyTorch / TensorFlow backends."""

    _cached_torch_device: Optional[Any] = None

    @classmethod
    def get_torch_device(cls, preferred: Optional[str] = None) -> Any:
        """
        Get appropriate torch.device based on availability and preference.
        
        Args:
            preferred: 'auto', 'cuda', 'mps', 'cpu', or None (uses DEVICE env var or 'auto')
            
        Returns:
            torch.device instance (or 'cpu' fallback if torch is unavailable)
        """
        if not HAS_TORCH:
            logger.warning("⚠️ PyTorch 未安裝，預設使用 CPU")
            return "cpu"

        pref = (preferred or os.getenv("DEVICE", "auto")).strip().lower()

        # Check explicit preference
        if pref == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            logger.warning("⚠️ 指定了 CUDA 但系統未檢測到可用 GPU，降級為 CPU")
            return torch.device("cpu")

        if pref == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            logger.warning("⚠️ 指定了 MPS 但系統不支援 Apple Silicon MPS，降級為 CPU")
            return torch.device("cpu")

        if pref == "cpu":
            return torch.device("cpu")

        # Auto detection
        if torch.cuda.is_available():
            try:
                # Test tensor creation
                test_tensor = torch.zeros(1, device="cuda")
                del test_tensor
                return torch.device("cuda")
            except Exception as e:
                logger.warning(f"⚠️ CUDA 測試失敗: {e}，降級為 CPU")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                test_tensor = torch.zeros(1, device="mps")
                del test_tensor
                return torch.device("mps")
            except Exception as e:
                logger.warning(f"⚠️ MPS 測試失敗: {e}，降級為 CPU")

        return torch.device("cpu")

    @classmethod
    def get_device_info(cls, device: Optional[Any] = None) -> dict:
        """
        Get detailed hardware and backend diagnostic information.
        """
        dev = device or cls.get_torch_device()
        dev_type = getattr(dev, "type", str(dev))
        
        info = {
            "device": str(dev),
            "type": dev_type,
            "has_torch": HAS_TORCH,
            "cuda_available": torch.cuda.is_available() if HAS_TORCH else False,
            "mps_available": (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) if HAS_TORCH else False,
            "name": "CPU"
        }

        if HAS_TORCH:
            if dev_type == "cuda" and torch.cuda.is_available():
                info["name"] = f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
            elif dev_type == "mps":
                info["name"] = "Apple Silicon (MPS)"
            else:
                info["name"] = "CPU"

        return info
