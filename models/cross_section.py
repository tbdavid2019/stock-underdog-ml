"""
Cross-sectional models for stock prediction (TabNet, SFM, ADDModel)
Requires qlib library
"""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import inspect
import importlib
from typing import List, Tuple, Optional, Any


# Cross-sectional model configurations
CROSS_MODELS = [
    ('qlib.contrib.model.pytorch_tabnet', ['TabNet', 'TabnetModel'])
    # ('qlib.contrib.model.pytorch_sfm',    ['SFM', 'SFMModel']),
    # ('qlib.contrib.model.pytorch_add',    ['ADDModel'])
]


def build_cross_xy(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    """
    Build features and labels for cross-sectional models
    Label = next day price change percentage = (Close(t+1) - Close(t)) / Close(t)
    
    Args:
        df: DataFrame with stock data including technical indicators
    
    Returns:
        Tuple of (X, y, meta) where:
            X: Feature tensor
            y: Label tensor (percentage returns)
            meta: Metadata with Date and Ticker
    """
    df = df.copy()
    
    # Create percentage return label
    df['pct_ret1'] = (
        df.groupby('Ticker')['Close'].transform(lambda x: x.shift(-1)) - df['Close']
    ) / df['Close']
    
    df = df.dropna()  # Remove last day without label
    feats = ['Open', 'High', 'Low', 'Close', 'Volume', 'ma5', 'ma10', 'rsi14']
    
    X = torch.tensor(df[feats].values, dtype=torch.float32)
    y = torch.tensor(df['pct_ret1'].values, dtype=torch.float32)
    meta = df[['Date', 'Ticker']].reset_index(drop=True)
    
    return X, y, meta


def import_model(mod_path: str, cls_list: List[str]) -> Optional[Any]:
    """
    Dynamically import model class from qlib
    
    Args:
        mod_path: Module path (e.g., 'qlib.contrib.model.pytorch_tabnet')
        cls_list: List of possible class names
    
    Returns:
        Model class or None if import fails
    """
    try:
        m = importlib.import_module(mod_path)
        for c in cls_list:
            if hasattr(m, c):
                return getattr(m, c)
    except ImportError:
        pass
    return None


def train_cross_loop(model_cls: Any, X: torch.Tensor, y: torch.Tensor, epochs: int, device: str = "cpu") -> np.ndarray:
    """
    Generic training loop for cross-sectional models
    Supports TabNet / SFM / ADDModel - auto-detects required __init__ parameters
    
    Args:
        model_cls: Model class to instantiate
        X: Feature tensor
        y: Label tensor
        epochs: Number of training epochs
        device: Device to use ("cpu" or "cuda")
    
    Returns:
        Array of predictions
    """
    sig = inspect.signature(model_cls.__init__)
    param_names = sig.parameters
    
    kw = {}
    # Feature input dimension
    if 'd_feat' in param_names:
        kw['d_feat'] = X.shape[1]
    if 'feature_dim' in param_names:
        kw['feature_dim'] = X.shape[1]
    if 'input_dim' in param_names:
        kw['input_dim'] = X.shape[1]
    if 'field_dim' in param_names:
        kw['field_dim'] = X.shape[1]
    
    # Output dimension
    if 'output_dim' in param_names:
        kw['output_dim'] = 1
    if 'target_dim' in param_names:
        kw['target_dim'] = 1
    if 'embed_dim' in param_names:
        kw['embed_dim'] = 16  # For SFM
    
    model = model_cls(**kw)
    net = model.model if hasattr(model, 'model') else model
    
    # Move to device if supported
    if hasattr(net, 'to'):
        net.to(device)
    
    ds = DataLoader(
        TensorDataset(X.to(device), y.to(device)),
        batch_size=512,
        shuffle=True
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    net.train()
    for _ in range(epochs):
        for xb, yb in ds:
            opt.zero_grad()
            out = net(xb)
            out = out[0] if isinstance(out, tuple) else out
            loss_fn(out.squeeze(), yb).backward()
            opt.step()
    
    net.eval()
    with torch.no_grad():
        preds = net(X.to(device))
        preds = preds[0] if isinstance(preds, tuple) else preds
        preds = preds.squeeze().cpu().numpy()
    
    return preds


def train_tabnet(X: torch.Tensor, y: torch.Tensor, epochs: int = 150, device: str = "cpu") -> np.ndarray:
    """
    Correct TabNet training loop with priors tensor
    
    Args:
        X: Feature tensor
        y: Label tensor
        epochs: Number of epochs
        device: Device to use
    
    Returns:
        Array of predictions
    """
    from qlib.contrib.model.pytorch_tabnet import TabNet
    
    inp = X.shape[1]
    net = TabNet(inp_dim=inp, out_dim=1).to(device)
    
    loader = DataLoader(
        TensorDataset(X.to(device), y.to(device)),
        batch_size=512,
        shuffle=True
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    net.train()
    for _ in range(epochs):
        for xb, yb in loader:
            pri = torch.ones(xb.size(0), inp, device=device)  # All-ones mask
            opt.zero_grad()
            raw = net(xb, priors=pri)  # May return tuple
            out = raw[0] if isinstance(raw, tuple) else raw
            loss_fn(out.squeeze(), yb).backward()
            opt.step()
    
    net.eval()
    with torch.no_grad():
        pri_all = torch.ones(X.size(0), inp, device=device)
        raw_all = net(X.to(device), priors=pri_all)
        preds = (raw_all[0] if isinstance(raw_all, tuple) else raw_all).squeeze().cpu().numpy()
    
    return preds
