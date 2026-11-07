"""
Sovereign AI Stack
====================
Master Blueprint: Sistem Komputasi Masa Depan

Modul:
    - TEG_Module       : Temporal Entropy Gating (filter input)
    - REA_Module       : Recursive Equilibrium Adaptation (reasoning)
    - DGA_Module       : Directional Gated Adaptation (task adaptation)
    - AON_Verifier     : Asymmetric Oracle Network (verification)
    - PTT_Optimizer    : Predictive Tensor Teleportation (GPU communication)
    - NLSI_Module      : Non-Local State Insulation (quantum isolation)
    - SovereignAIManager : Orchestrator (pusat kendali)
    - BenchmarkRunner  : Framework pengujian performa
    - BenchmarkDashboard : Visualisasi real-time
"""

from .TEG_Module import TEG_Module
from .REA_Module import REA_Module
from .DGA_Module import DGA_Module
from .AON_Verifier import AON_Verifier
from .PTT_Optimizer import PTT_Optimizer
from .NLSI_Module import NLSI_Module
from .SovereignAIManager import SovereignAIManager
from .BenchmarkRunner import BenchmarkRunner
from .BenchmarkDashboard import BenchmarkDashboard

__version__ = "1.0.0"
__all__ = [
    'TEG_Module',
    'REA_Module',
    'DGA_Module',
    'AON_Verifier',
    'PTT_Optimizer',
    'NLSI_Module',
    'SovereignAIManager',
    'BenchmarkRunner',
    'BenchmarkDashboard',
]
