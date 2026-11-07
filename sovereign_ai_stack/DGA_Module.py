"""
Directional Gated Adaptation (DGA)
====================================
Modul adaptasi tugas spesifik (Task Specialist).
Memungkinkan AI berpindah mode (spesialisasi) secara instan tanpa retrain.

PRODUCTION READY v2.0
- Input validation & type hints
- NaN/Inf detection in adapter output
- Gate value clamping for stability
- Proper logging
- Multiple adapter support (future: swap adapters)

Konsep:
    Bobot inti (REA) dibekukan (freeze), hanya jalur samping yang dilatih.
    Jalur samping memiliki "Gerbang" (Gate) yang mengatur pengaruh spesialis.

Rumus:
    Output = Base(x) + Gate(Task) ⊙ Adapter(x)
    
    Dimana:
    - Base(x): Logika fundamental dari REA
    - Adapter(x): Pengetahuan spesialis (Medis/Hukum/Coding)
    - Gate: Pengatur "seberapa banyak" spesialisasi yang digunakan
    - ⊙ : Elemen-wise multiplication
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DGA_Module(nn.Module):
    """
    Directional Gated Adaptation (DGA)

    Memungkinkan AI berpindah mode (spesialisasi) secara instan.
    Menggunakan Low-Rank Adaptation + Gating Mechanism.

    Args:
        d_model (int): Dimensi model (embedding size)
        adapter_dim (int): Dimensi adapter low-rank (default: 64, min: 4, max: 2048)

    Raises:
        ValueError: Jika parameter tidak valid
    """

    def __init__(self, d_model: int, adapter_dim: int = 64):
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if adapter_dim < 4:
            raise ValueError(f"adapter_dim must be >= 4, got {adapter_dim}")
        if adapter_dim > 2048:
            raise ValueError(f"adapter_dim must be <= 2048, got {adapter_dim}")

        # Down-projection & Up-projection (Low-rank adaptation)
        # Menekan jumlah parameter drastis dengan bottleneck
        self.down_proj = nn.Linear(d_model, adapter_dim, bias=False)
        self.up_proj = nn.Linear(adapter_dim, d_model, bias=False)
        self.activation = nn.ReLU()

        # Gating Mechanism: Menentukan pengaruh modul spesialis
        # Parameter yang bisa dipelajari, diinisialisasi 0
        # Awalnya adapter tidak berpengaruh, belajar secara bertahap
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass dengan residual gating.

        Args:
            x (torch.Tensor): Output tensor dari REA [Batch, Seq_Len, D_Model]

        Returns:
            torch.Tensor: Output dengan adaptasi spesialis

        Raises:
            ValueError: Jika input tidak valid
        """
        self._validate_input(x)

        # Basis (REA) tetap dominan
        # Adapter memberikan "warna" spesialisasi
        adapter_out = self.up_proj(self.activation(self.down_proj(x)))

        # Safety: cek NaN di adapter output
        if torch.isnan(adapter_out).any() or torch.isinf(adapter_out).any():
            logger.warning("NaN/Inf detected in DGA adapter, clamping...")
            adapter_out = torch.nan_to_num(adapter_out, nan=0.0, posinf=1e6, neginf=-1e6)

        # Clamp gate to prevent extreme adaptation
        gate_val = torch.clamp(self.gate, min=-1.0, max=1.0)

        return x + (gate_val * adapter_out)

    def set_adapter_dim(self, new_dim: int) -> None:
        """
        Ubah dimensi adapter (re-initialize).
        Berguna untuk kompresi/dekompresi model.

        Args:
            new_dim (int): Dimensi adapter baru
        """
        d_model = self.down_proj.in_features
        device = self.down_proj.weight.device

        self.down_proj = nn.Linear(d_model, new_dim, bias=False).to(device)
        self.up_proj = nn.Linear(new_dim, d_model, bias=False).to(device)

        logger.info(f"DGA adapter dim changed: {self.down_proj.out_features} → {new_dim}")

    def _validate_input(self, x: torch.Tensor) -> None:
        """
        Validasi input tensor.

        Args:
            x (torch.Tensor): Input tensor

        Raises:
            ValueError: Jika input tidak valid
        """
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(x).__name__}")

        if x.numel() == 0:
            raise ValueError("Empty tensor received")

        if x.dim() < 2 or x.dim() > 3:
            raise ValueError(
                f"Expected 2D or 3D tensor, got {x.dim()}D"
            )

        if x.shape[-1] != self.down_proj.in_features:
            raise ValueError(
                f"Last dimension {x.shape[-1]} doesn't match d_model={self.down_proj.in_features}"
            )

    def extra_repr(self) -> str:
        """Representasi tambahan untuk print(model)"""
        return (
            f"d_model={self.down_proj.in_features}, "
            f"adapter_dim={self.down_proj.out_features}"
        )
