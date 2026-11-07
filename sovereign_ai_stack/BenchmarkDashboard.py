"""
Benchmark Dashboard
=====================
Dashboard visualisasi real-time untuk memantau performa Sovereign AI Stack.
PRODUCTION READY v2.0 dengan error handling, auto-refresh yang aman, dan CLI mode.

Visualisasi mencakup:
    1. PTT Communication Efficiency (Gradient Sparsity)
    2. AON Verification Success Rate
    3. Energy Efficiency (TFLOPS/Watt)
    4. Inference Latency
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import matplotlib, provide graceful fallback
try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not installed. Dashboard will run in CLI mode.")


class BenchmarkDashboard:
    """
    Dashboard monitoring real-time untuk Sovereign AI Stack.

    Menampilkan 4 grafik performa secara live:
    - Top-Left: PTT Communication Efficiency
    - Top-Right: AON Verification Success Rate
    - Bottom-Left: Energy Efficiency
    - Bottom-Right: Inference Latency

    Args:
        use_gui (bool): Jika True, tampilkan GUI. Jika False, CLI mode. (default: auto)
    """

    def __init__(self, use_gui: Optional[bool] = None):
        # Auto-detect GUI availability
        self.use_gui = use_gui if use_gui is not None else MATPLOTLIB_AVAILABLE

        self.history: Dict[str, List[float]] = {
            'sparsity': [],
            'throughput': [],
            'efficiency': [],
            'latency': []
        }

        if self.use_gui and MATPLOTLIB_AVAILABLE:
            try:
                plt.ion()  # Mode interaktif
                self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 8))
                logger.info("Dashboard GUI initialized")
            except Exception as e:
                logger.warning("Failed to initialize GUI: %s. Falling back to CLI mode.", e)
                self.use_gui = False
        else:
            logger.info("Dashboard running in CLI mode")

    def update(self, metrics: Dict[str, float]) -> None:
        """
        Perbarui dashboard dengan data performa terbaru.

        Args:
            metrics (dict): Data performa dengan keys:
                - sparsity: Rasio sparsity PTT
                - throughput: Tingkat verifikasi AON (%)
                - efficiency: Efisiensi energi (TFLOPS/Watt)
                - latency: Latensi inferensi (ms)

        Raises:
            ValueError: Jika metrics tidak lengkap
        """
        required_keys = ['sparsity', 'throughput', 'efficiency', 'latency']
        for key in required_keys:
            if key not in metrics:
                raise ValueError(f"Missing required metric: {key}")

        # Append to history
        self.history['sparsity'].append(metrics['sparsity'])
        self.history['throughput'].append(metrics['throughput'])
        self.history['efficiency'].append(metrics['efficiency'])
        self.history['latency'].append(metrics['latency'])

        if self.use_gui and MATPLOTLIB_AVAILABLE:
            self._update_gui(metrics)
        else:
            self._update_cli(metrics)

    def _update_gui(self, metrics: Dict[str, float]) -> None:
        """
        Update GUI matplotlib plot.

        Args:
            metrics (dict): Data performa
        """
        try:
            # Plot 1: PTT Gradient Sparsity (Efisiensi Jaringan)
            self.axs[0, 0].clear()
            self.axs[0, 0].plot(self.history['sparsity'], label='Data Komunikasi (Sparsity)')
            self.axs[0, 0].set_title('PTT Communication Efficiency')
            self.axs[0, 0].set_ylabel('% Data Dikirim')
            self.axs[0, 0].legend()
            self.axs[0, 0].grid(True, alpha=0.3)

            # Plot 2: AON Verification Throughput (Ketepatan Logika)
            self.axs[0, 1].clear()
            verified = metrics['throughput']
            rejected = max(0, 100 - verified)
            bars = self.axs[0, 1].bar(['Verified', 'Rejected'], [verified, rejected],
                                     color=['green', 'red'])
            self.axs[0, 1].set_title('AON Verification Success Rate')
            self.axs[0, 1].set_ylabel('Persentase (%)')
            self.axs[0, 1].set_ylim([0, 100])

            # Add value labels on bars
            for bar, val in zip(bars, [verified, rejected]):
                height = bar.get_height()
                self.axs[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{val:.1f}%', ha='center', va='bottom')

            # Plot 3: Energy Efficiency (TFLOPS/Watt)
            self.axs[1, 0].clear()
            self.axs[1, 0].plot(self.history['efficiency'], color='green', linewidth=2)
            self.axs[1, 0].set_title('Energy Efficiency (TFLOPS/Watt)')
            self.axs[1, 0].set_ylabel('TFLOPS/Watt')
            self.axs[1, 0].grid(True, alpha=0.3)

            # Plot 4: Inference Latency
            self.axs[1, 1].clear()
            self.axs[1, 1].plot(self.history['latency'], color='red', linewidth=2)
            self.axs[1, 1].set_title('Inference Latency (ms)')
            self.axs[1, 1].set_ylabel('Milidetik (ms)')
            self.axs[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.draw()
            plt.pause(0.05)  # Non-blocking with minimal delay

        except Exception as e:
            logger.error("Dashboard GUI update failed: %s", e)

    def _update_cli(self, metrics: Dict[str, float]) -> None:
        """
        Update CLI text output.

        Args:
            metrics (dict): Data performa
        """
        n = len(self.history['sparsity'])
        print(f"\n{'='*55}")
        print(f"📊 Benchmark Dashboard [Update #{n}]")
        print(f"{'='*55}")
        print(f"  📡 PTT Sparsity:       {metrics['sparsity']*100:6.2f}% "
              f"(compression: 1:{1/max(metrics['sparsity'], 0.01):.0f})")
        print(f"  ✅ AON Verification:   {metrics['throughput']:6.2f}%")
        print(f"  ⚡ Energy Efficiency:  {metrics['efficiency']:6.2f} TFLOPS/Watt")
        print(f"  ⏱️  Inference Latency: {metrics['latency']:6.2f} ms")

    def get_summary(self) -> Dict[str, float]:
        """
        Dapatkan ringkasan statistik dari seluruh history.

        Returns:
            dict: Statistik summary (mean, min, max untuk setiap metrik)
        """
        summary = {}
        for key, values in self.history.items():
            if values:
                summary[f"{key}_mean"] = sum(values) / len(values)
                summary[f"{key}_min"] = min(values)
                summary[f"{key}_max"] = max(values)
                summary[f"{key}_latest"] = values[-1]
            else:
                summary[f"{key}_mean"] = 0.0
                summary[f"{key}_min"] = 0.0
                summary[f"{key}_max"] = 0.0
                summary[f"{key}_latest"] = 0.0
        return summary

    def close(self) -> None:
        """Tutup dashboard dan bersihkan resource."""
        if self.use_gui and MATPLOTLIB_AVAILABLE:
            try:
                plt.ioff()
                plt.close('all')
                logger.info("Dashboard GUI closed")
            except Exception as e:
                logger.warning("Error closing dashboard: %s", e)


# Contoh simulasi penggunaan
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dashboard = BenchmarkDashboard()

    import numpy as np

    # Simulasi aliran data real-time dari Sovereign AI Stack
    for i in range(50):
        mock_data = {
            'sparsity': max(0.01, 1.0 + np.sin(i * 0.1) * 0.5),  # PTT memangkas data dinamis
            'throughput': min(100, max(0, 85 + np.random.randn() * 5)),  # AON jaga akurasi tinggi
            'efficiency': 120 + i * 0.5,  # Efisiensi naik bertahap
            'latency': max(1, 15 - (i * 0.1) + np.random.randn() * 0.5),  # Latensi turun krn optimasi
        }
        dashboard.update(mock_data)

    print("\n📊 Final Summary:")
    summary = dashboard.get_summary()
    for key, val in summary.items():
        print(f"  {key}: {val:.2f}")

    dashboard.close()
