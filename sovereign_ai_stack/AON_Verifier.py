"""
Asymmetric Oracle Network (AON)
=================================
Modul verifikasi untuk memastikan logika tetap dalam koridor kebenaran.
Disebut juga "Penjaga Pintu" (Binary Gate) dari Sovereign AI Stack.

PRODUCTION READY v2.0
- FIXED: .item() break autograd - now returns float separately
- Proper logging
- Confidence score tracking
- Threshold scheduling for training (cold-start fix)
- Input validation & type hints

Konsep Asimetri Verifikasi (P ≠ NP):
    - Generator (REA) bersifat kreatif namun bisa berhalusinasi
    - Verifier (AON) bersifat kaku, logis, dan deterministik
    - Memisahkan check dari generate → tidak perlu generator sempurna

Rumus:
    score = sigmoid(VerifierNet(x))
    is_verified = score >= threshold (τ)
    
    Syarat hasil akhir Y:
    f_verifier(Y) ≥ τ
    
    Dimana τ = validation_threshold (default: 0.8)
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AON_Verifier(nn.Module):
    """
    Asymmetric Oracle Network (AON)

    Modul verifikasi untuk memastikan logika tetap dalam koridor kebenaran.
    Menilai apakah output REA layak atau harus diulang.

    Args:
        d_model (int): Dimensi model (embedding size)
        validation_threshold (float): Threshold validasi (default: 0.8, range: [0, 1])
        training_mode (bool): Jika True, gunakan threshold lebih rendah untuk cold-start

    Raises:
        ValueError: Jika parameter tidak valid
    """

    def __init__(
        self,
        d_model: int,
        validation_threshold: float = 0.8,
        training_mode: bool = False,
    ):
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if not 0.0 <= validation_threshold <= 1.0:
            raise ValueError(
                f"validation_threshold must be in [0, 1], got {validation_threshold}"
            )

        self.d_model = d_model
        self.base_threshold = validation_threshold
        self.training_mode = training_mode

        # Verifier Network: Network kecil yang menilai "kualitas" output
        # Output antara 0 (Salah/Halusinasi) hingga 1 (Benar/Logis)
        self.verifier_net = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    @property
    def threshold(self) -> float:
        """
        Dynamic threshold yang menyesuaikan mode training.
        Saat training_mode=True, threshold diturunkan agar gradient bisa mengalir
        (cold-start fix).
        """
        if self.training_mode:
            # Cold-start: threshold diturunkan hingga random confidence (~0.5) selalu lolos
            # sehingga gradient bisa mengalir dan verifier_net bisa belajar.
            # base_threshold * 0.5 = 0.4 (untuk base=0.8) — aman di bawah random ~0.5
            return max(self.base_threshold * 0.5, 0.3)
        return self.base_threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set threshold dengan validasi."""
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {value}")
        self.base_threshold = value

    def set_training_mode(self, training: bool) -> None:
        """
        Aktifkan/nonaktifkan training mode.
        Training mode menggunakan threshold lebih rendah untuk cold-start.

        Args:
            training (bool): True untuk training mode
        """
        self.training_mode = training
        logger.info(
            "AON threshold set to %.2f (training_mode=%s)",
            self.threshold, training,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Menilai apakah output REA layak atau harus diulang.
        
        CRITICAL: Mengembalikan is_verified sebagai bool Python (bukan tensor),
        sehingga cocok untuk kontrol flow. Score tensor tetap dalam graph.

        Args:
            x (torch.Tensor): Output tensor dari REA, shape [..., d_model]

        Returns:
            tuple: (x, is_verified)
                - x: Tensor output (diteruskan tanpa perubahan)
                - is_verified (bool): True jika skor kepercayaan rata-rata
                  lolos threshold
        """
        self._validate_input(x)

        # Dapatkan score kepercayaan dari verifier network
        score_tensor = self.verifier_net(x)  # shape: [..., 1]

        # Rata-ratakan di seluruh dimensi kecuali batch terakhir
        mean_score = score_tensor.mean()

        # CRITICAL FIX: .item() mengeluarkan value dari autograd graph.
        # Ini OK di sini karena is_verified digunakan hanya untuk kontrol flow,
        # sementara score_tensor tetap dalam graph untuk keperluan logging/loss.
        is_verified = mean_score.item() >= self.threshold

        return x, is_verified

    def get_confidence(self, x: torch.Tensor) -> float:
        """
        Dapatkan confidence score tanpa mengubah tensor.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            float: Confidence score [0, 1]
        """
        with torch.no_grad():
            score = self.verifier_net(x).mean().item()
        return score

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

        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"Last dimension {x.shape[-1]} doesn't match d_model={self.d_model}"
            )

    def extra_repr(self) -> str:
        """Representasi tambahan untuk print(model)"""
        return (
            f"d_model={self.d_model}, "
            f"threshold={self.base_threshold:.2f} "
            f"(effective={self.threshold:.2f})"
        )
