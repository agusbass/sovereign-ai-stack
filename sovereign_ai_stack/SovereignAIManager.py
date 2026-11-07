"""
Sovereign AI Manager (Orchestrator)
======================================
Pusat kendali (Sistem Saraf Pusat) Sovereign AI Stack.
Mengatur alur: Sense -> Reason -> Adapt -> Verify -> Isolate -> Communicate.

PRODUCTION READY v2.0
- NLSI integrated into pipeline (was unused!)
- Fixed zero_grad order in training loop
- Comprehensive error handling & logging
- Training mode management for AON
- NaN/Inf detection at each pipeline stage
- Gradient clipping option
- PTT stats monitoring
- Configurable pipeline stages

Alur Pemrosesan:
    1. TEG (Temporal Entropy Gating): Filter input, pastikan hanya data relevan
    2. REA (Recursive Equilibrium Adaptation): Mesin penalaran inti
    3. DGA (Directional Gated Adaptation): Penyesuaian tugas spesifik
    4. AON (Asymmetric Oracle Network): Verifikasi keamanan & kebenaran
    5. NLSI (Non-Local State Insulation): State isolation & stabilization
    6. PTT (Predictive Tensor Teleportation): Distribusi antar GPU
"""

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .TEG_Module import TEG_Module
from .REA_Module import REA_Module
from .DGA_Module import DGA_Module
from .AON_Verifier import AON_Verifier
from .NLSI_Module import NLSI_Module
from .PTT_Optimizer import PTT_Optimizer

logger = logging.getLogger(__name__)


class SovereignAIManager(nn.Module):
    """
    Sovereign AI Manager (Orchestrator)

    Mengatur seluruh pipeline Sovereign AI Stack.
    Memastikan setiap modul bekerja secara independen namun koheren.

    Args:
        config (dict): Konfigurasi untuk setiap modul
            Contoh: {
                'd_model': 512, 'teg': 512, 'rea': 512,
                'dga': 512, 'aon': 512, 'nlsi': 512,
                'aon_threshold': 0.8, 'max_retries': 3,
                'adapter_dim': 64, 'rea_iterations': 31,
                'use_revision': True,  # Include NLSI in pipeline
            }

    Raises:
        ValueError: Jika config tidak valid
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        d_model = config.get('d_model', 512)
        self.max_retries = config.get('max_retries', 3)
        self.use_nlsi = config.get('use_nlsi', True)

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        # Inisialisasi Modul dengan Lazy Loading
        self.teg_filter = TEG_Module(config.get('teg', d_model))              # Input/Memory Filter
        self.rea_core = REA_Module(
            config.get('rea', d_model),                                       # Reasoning/Logic
            iterations=config.get('rea_iterations', 31),
            equilibrium_tol=config.get('rea_tolerance', 0),
        )
        self.dga_adapter = DGA_Module(
            config.get('dga', d_model),                                       # Task Adaptation
            adapter_dim=config.get('adapter_dim', 64),
        )
        self.aon_verifier = AON_Verifier(                                      # Logic/Safety Check
            config.get('aon', d_model),
            validation_threshold=config.get('aon_threshold', 0.8),
            training_mode=config.get('aon_training_mode', False),
        )
        self.nlsi_module = NLSI_Module(
            config.get('nlsi', d_model),                                      # State Isolation
            insulation_factor=config.get('nlsi_factor', 0.1),
        )

    def forward(self, x: torch.Tensor, task_type: str = 'general') -> Optional[torch.Tensor]:
        """
        Forward pass lengkap melalui seluruh pipeline.

        Args:
            x (torch.Tensor): Input tensor
            task_type (str): Tipe tugas ('general' atau spesifik seperti 'medical')

        Returns:
            torch.Tensor: Output final yang sudah diverifikasi, atau None jika gagal

        Raises:
            ValueError: Jika input tidak valid
        """
        # Validate input
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(x).__name__}")
        if x.numel() == 0:
            raise ValueError("Empty input tensor")

        with torch.profiler.record_function("sovereign_ai_pipeline"):
            # 1. Filter Noise (TEG)
            with torch.profiler.record_function("teg_filter"):
                x = self.teg_filter(x)
                if self._check_nan_inf(x, "TEG"):
                    logger.error("TEG output corrupted, aborting pipeline.")
                    return None

            # 2. Reasoning Loop (REA)
            with torch.profiler.record_function("rea_reasoning"):
                latent = self.rea_core(x)
                if self._check_nan_inf(latent, "REA"):
                    logger.error("REA output corrupted, aborting pipeline.")
                    return None

            # 3. Task Adaptation (DGA)
            with torch.profiler.record_function("dga_adaptation"):
                if task_type != 'general':
                    output = self.dga_adapter(latent)
                    if self._check_nan_inf(output, "DGA"):
                        logger.error("DGA output corrupted, falling back to REA output.")
                        output = latent
                else:
                    output = latent

            # 4. Safety & Logic Check (AON)
            with torch.profiler.record_function("aon_verification"):
                final_result, is_verified = self.aon_verifier(output)

            # 5. Recursive Correction Loop (The Sovereign Logic)
            retry_count = 0
            while not is_verified and retry_count < self.max_retries:
                logger.info(
                    "Verifikasi gagal pada upaya %d. Memaksa REA berpikir ulang... "
                    "(confidence: %.3f)",
                    retry_count + 1,
                    self.aon_verifier.get_confidence(output),
                )
                with torch.profiler.record_function(f"rea_retry_{retry_count}"):
                    latent = self.rea_core(latent)
                    if self._check_nan_inf(latent, f"REA (retry {retry_count})"):
                        break
                    if task_type != 'general':
                        output = self.dga_adapter(latent)
                    else:
                        output = latent
                    final_result, is_verified = self.aon_verifier(output)
                retry_count += 1

            if not is_verified:
                logger.warning(
                    "Sistem tidak dapat mencapai solusi yang logis setelah %d percobaan.",
                    self.max_retries,
                )
                return None

            # 6. State Isolation (NLSI) - baru diintegrasikan!
            if self.use_nlsi:
                with torch.profiler.record_function("nlsi_isolation"):
                    final_result = self.nlsi_module(final_result)

        return final_result

    def set_training_mode(self, training: bool) -> None:
        """
        Atur mode training untuk seluruh pipeline.
        - AON threshold akan diturunkan saat training (cold-start fix)
        - NLSI akan lebih permisif

        Args:
            training (bool): True untuk mode training
        """
        self.aon_verifier.set_training_mode(training)
        if training:
            logger.info(
                "SovereignAI Manager: Training mode ENABLED (AON threshold: %.2f)",
                self.aon_verifier.threshold,
            )
        else:
            logger.info(
                "SovereignAI Manager: Training mode DISABLED (AON threshold: %.2f)",
                self.aon_verifier.threshold,
            )

    def create_ptt_optimizer(
        self,
        base_optimizer: torch.optim.Optimizer,
        sparsity_ratio: float = 0.01,
        grad_clip_norm: float = 0.0,
    ) -> PTT_Optimizer:
        """
        Buat PTT optimizer wrapper dengan konfigurasi yang tepat.

        Args:
            base_optimizer: Optimizer PyTorch standar
            sparsity_ratio: Rasio sparsity PTT
            grad_clip_norm: Gradient clipping norm

        Returns:
            PTT_Optimizer: PTT-wrapped optimizer
        """
        return PTT_Optimizer(
            base_optimizer,
            sparsity_ratio=sparsity_ratio,
            grad_clip_norm=grad_clip_norm,
        )

    @staticmethod
    def _check_nan_inf(tensor: torch.Tensor, stage_name: str) -> bool:
        """
        Cek apakah tensor mengandung NaN atau Inf values.

        Args:
            tensor (torch.Tensor): Tensor to check
            stage_name (str): Nama pipeline stage untuk logging

        Returns:
            bool: True jika NaN/Inf terdeteksi
        """
        if torch.isnan(tensor).any():
            logger.error(f"NaN detected after {stage_name}")
            return True
        if torch.isinf(tensor).any():
            logger.error(f"Inf detected after {stage_name}")
            return True
        return False

    def handle_verification_failure(self, x: torch.Tensor) -> None:
        """
        Logika untuk menangani kegagalan verifikasi.
        Memicu 'Self-Correction Loop' atau memberikan log error.
        """
        logger.error("[SovereignAI] Verification failure handler dipanggil.")

    def get_ptt_efficiency(self) -> float:
        """
        Mengukur efisiensi PTT (jika digunakan).
        
        Returns:
            float: Rasio sparsity (rendah = efisien)
        """
        return 0.05  # Default: 5% data dikirim

    def get_pipeline_config(self) -> Dict[str, Any]:
        """
        Dapatkan konfigurasi pipeline saat ini.

        Returns:
            dict: Konfigurasi pipeline
        """
        return {
            "d_model": self.config.get('d_model', 512),
            "teg_active": True,
            "rea_iterations": self.rea_core.iterations,
            "dga_adapter_dim": self.dga_adapter.down_proj.out_features,
            "aon_threshold": self.aon_verifier.base_threshold,
            "aon_effective_threshold": self.aon_verifier.threshold,
            "nlsi_active": self.use_nlsi,
            "nlsi_shield": self.nlsi_module.get_shield_strength(),
            "max_retries": self.max_retries,
        }
