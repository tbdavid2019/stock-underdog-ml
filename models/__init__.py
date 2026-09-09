"""
Stock prediction models package
"""
from . import lstm
try:
    from . import timesfm_model
except Exception:
    pass

__all__ = ['lstm', 'timesfm_model']
