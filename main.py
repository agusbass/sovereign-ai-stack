"""
Main.py - Contoh Training Loop Sovereign AI Stack (PRODUCTION READY)
======================================================================
Menunjukkan bagaimana semua modul bekerja bersama dalam satu training loop.
PRODUCTION READY v2.0 dengan training mode, error handling, dan PTT monitoring.
"""

import logging
import sys

import torch
import torch.nn as nn

from sovereign_ai_stack import SovereignAIManager, PTT_Optimizer

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    """
    Contoh penggunaan Sovereign AI Stack dalam training loop
    dengan production-ready practices.
    """
    # 1. Konfigurasi Production-Ready
    config = {
        'd_model': 512,
        'teg': 512,
        'rea': 512,
        'dga': 512,
        'aon': 512,
        'nlsi': 512,
        # AON_Verifier: Training mode otomatis menurunkan threshold
        # agar gradient bisa mengalir (cold-start fix)
        'aon_threshold': 0.8,
        'aon_training_mode': True,  # Threshold akan diturunkan otomatis
        'max_retries': 3,
        'adapter_dim': 64,
        'rea_iterations': 31,
        'use_nlsi': True,  # NLSI sekarang terintegrasi penuh!
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Menggunakan device: %s", device)

    # 2. Inisialisasi Sovereign AI Stack
    logger.info("Initializing Sovereign AI Stack...")
    sovereign_ai = SovereignAIManager(config).to(device)

    # Set training mode (AON threshold akan diturunkan)
    sovereign_ai.set_training_mode(training=True)

    # 3. Buat PTT Optimizer (gradient clipping enabled)
    base_optimizer = torch.optim.AdamW(
        sovereign_ai.parameters(),
        lr=1e-3,
        weight_decay=0.01,
    )
    ptt_engine = sovereign_ai.create_ptt_optimizer(
        base_optimizer,
        sparsity_ratio=0.05,
        grad_clip_norm=1.0,  # Gradient clipping untuk stabilitas
    )

    # 4. Loss function
    criterion = nn.MSELoss()

    # 5. Dummy data (contoh)
    batch_size, seq_len, d_model = 4, 128, 512
    dummy_input = torch.randn(batch_size, seq_len, d_model, device=device)
    dummy_target = torch.randn(batch_size, seq_len, d_model, device=device)

    # 6. Training Loop (PRODUCTION READY)
    num_epochs = 10
    logger.info("Starting training for %d epochs...", num_epochs)

    for epoch in range(num_epochs):
        # FIXED: zero_grad SEBELUM forward (bukan sesudah PTT step)
        ptt_engine.zero_grad()

        # Forward Pass (melalui TEG → REA → DGA → AON → NLSI)
        output = sovereign_ai(dummy_input, task_type='general')

        if output is None:
            logger.info("Epoch %d: Output gagal verifikasi, skip...", epoch + 1)
            continue

        # Hitung loss
        loss = criterion(output, dummy_target)

        # Backward Pass
        loss.backward()

        # PTT Step (teleportasi gradien antar GPU)
        ptt_engine.step()

        # Logging dengan PTT stats
        ptt_stats = ptt_engine.get_stats()
        logger.info(
            "Epoch %d/%d | Loss: %.6f | "
            "Sparsity: %.2f%% | Cache: %d/%d | "
            "Compression: 1:%.0f | NaN clipped: %d",
            epoch + 1, num_epochs, loss.item(),
            ptt_stats["sparsity_ratio"] * 100,
            ptt_stats["cache_size"],
            ptt_stats["max_cache_size"],
            ptt_stats["compression_ratio"],
            ptt_stats["nan_clipped"],
        )

    # Pipeline config summary
    pipeline_cfg = sovereign_ai.get_pipeline_config()
    logger.info("=" * 55)
    logger.info("Pipeline Configuration Summary:")
    for key, val in pipeline_cfg.items():
        logger.info("  %s: %s", key, val)
    logger.info("=" * 55)
    logger.info("Training selesai!")


if __name__ == "__main__":
    main()
