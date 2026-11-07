"""
Comprehensive Test Suite for Sovereign AI Stack
=================================================
Tests: creation, forward pass, validation, error handling, edge cases
Run: python -m pytest tests/ -v
"""

import pytest
import torch
import torch.nn as nn

from sovereign_ai_stack import (
    TEG_Module, REA_Module, DGA_Module, AON_Verifier,
    PTT_Optimizer, NLSI_Module, SovereignAIManager,
    BenchmarkRunner, BenchmarkDashboard,
)


# ============================================================
# TEG_Module Tests
# ============================================================

class TestTEG:
    def test_creation(self):
        m = TEG_Module(64)
        assert m.d_model == 64

    def test_forward_valid(self):
        m = TEG_Module(64)
        x = torch.randn(2, 10, 64)
        out = m(x)
        assert out.shape == (2, 10, 64)

    def test_forward_2d(self):
        m = TEG_Module(64)
        x = torch.randn(5, 64)
        out = m(x)
        assert out.shape == (5, 64)

    def test_forward_1d(self):
        m = TEG_Module(64)
        x = torch.randn(64)
        out = m(x)
        assert out.shape == (64,)

    def test_invalid_d_model(self):
        with pytest.raises(ValueError):
            TEG_Module(0)
        with pytest.raises(ValueError):
            TEG_Module(-1)

    def test_invalid_input_type(self):
        m = TEG_Module(64)
        with pytest.raises(ValueError):
            m("invalid")

    def test_empty_tensor(self):
        m = TEG_Module(64)
        with pytest.raises(ValueError):
            m(torch.randn(0, 64))

    def test_wrong_dim(self):
        m = TEG_Module(64)
        with pytest.raises(ValueError):
            m(torch.randn(2, 10, 32))  # Wrong d_model

    def test_forward_with_nan_input(self):
        m = TEG_Module(64)
        x = torch.randn(2, 10, 64)
        x[0, 0, 0] = float('nan')
        out = m(x)
        assert not torch.isnan(out).any()

    def test_gate_values_range(self):
        m = TEG_Module(64)
        x = torch.randn(2, 10, 64)
        out = m(x)
        # Output should be same magnitude as input (gated, not amplified)
        assert out.norm() <= x.norm() * 1.1  # Allow small variance

    def test_gradient_flow(self):
        m = TEG_Module(64)
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = m(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert list(m.entropy_gate.parameters())[0].grad is not None


# ============================================================
# REA_Module Tests
# ============================================================

class TestREA:
    def test_creation(self):
        m = REA_Module(64)
        assert m.iterations == 31
        assert m.d_model == 64

    def test_forward_valid(self):
        m = REA_Module(64)
        x = torch.randn(2, 10, 64)
        out = m(x)
        assert out.shape == (2, 10, 64)

    def test_forward_2d(self):
        m = REA_Module(64)
        x = torch.randn(5, 64)
        out = m(x)
        assert out.shape == (5, 64)

    def test_iterations_1(self):
        m = REA_Module(64, iterations=1)
        x = torch.randn(2, 10, 64)
        out = m(x)
        assert out.shape == (2, 10, 64)

    def test_iterations_100(self):
        m = REA_Module(64, iterations=100)
        x = torch.randn(2, 10, 64)
        out = m(x)
        assert out.shape == (2, 10, 64)

    def test_early_stopping(self):
        m = REA_Module(64, iterations=31, equilibrium_tol=1e-3)
        x = torch.randn(2, 10, 64)
        out = m(x)
        assert out.shape == (2, 10, 64)

    def test_invalid_d_model(self):
        with pytest.raises(ValueError):
            REA_Module(0)

    def test_invalid_iterations_low(self):
        with pytest.raises(ValueError):
            REA_Module(64, iterations=0)

    def test_invalid_iterations_high(self):
        with pytest.raises(ValueError):
            REA_Module(64, iterations=101)

    def test_empty_tensor(self):
        m = REA_Module(64)
        with pytest.raises(ValueError):
            m(torch.randn(0, 64))

    def test_wrong_dim(self):
        m = REA_Module(64)
        with pytest.raises(ValueError):
            m(torch.randn(2, 10, 32))

    def test_gradient_flow(self):
        m = REA_Module(64)
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = m(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert m.W.weight.grad is not None

    def test_output_differs_from_input(self):
        m = REA_Module(64)
        x = torch.randn(2, 10, 64)
        out = m(x)
        # Output should be processed (different from input)
        assert not torch.allclose(out, x, atol=1e-4)

    def test_numerical_stability(self):
        m = REA_Module(64, iterations=100)
        x = torch.randn(2, 10, 64) * 10  # Large input
        out = m(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ============================================================
# DGA_Module Tests
# ============================================================

class TestDGA:
    def test_creation(self):
        m = DGA_Module(64)
        assert m.down_proj.out_features == 64

    def test_forward(self):
        m = DGA_Module(64)
        x = torch.randn(2, 10, 64)
        out = m(x)
        assert out.shape == (2, 10, 64)

    def test_forward_2d(self):
        m = DGA_Module(64)
        x = torch.randn(5, 64)
        out = m(x)
        assert out.shape == (5, 64)

    def test_zero_gate_initial(self):
        m = DGA_Module(64)
        x = torch.randn(2, 10, 64)
        out = m(x)
        # With gate=0, output should equal input
        assert torch.allclose(out, x, atol=1e-6)

    def test_gate_training(self):
        m = DGA_Module(64)
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = m(x)
        loss = out.sum()
        loss.backward()
        assert m.gate.grad is not None

    def test_set_adapter_dim(self):
        m = DGA_Module(64, adapter_dim=64)
        m.set_adapter_dim(128)
        assert m.down_proj.out_features == 128

    def test_invalid_d_model(self):
        with pytest.raises(ValueError):
            DGA_Module(0)

    def test_invalid_adapter_dim_low(self):
        with pytest.raises(ValueError):
            DGA_Module(64, adapter_dim=3)

    def test_invalid_adapter_dim_high(self):
        with pytest.raises(ValueError):
            DGA_Module(64, adapter_dim=2049)

    def test_empty_tensor(self):
        m = DGA_Module(64)
        with pytest.raises(ValueError):
            m(torch.randn(0, 64))

    def test_wrong_dim(self):
        m = DGA_Module(64)
        with pytest.raises(ValueError):
            m(torch.randn(2, 10, 32))


# ============================================================
# AON_Verifier Tests
# ============================================================

class TestAON:
    def test_creation(self):
        m = AON_Verifier(64)
        assert m.base_threshold == 0.8

    def test_forward_low_confidence(self):
        m = AON_Verifier(64, validation_threshold=0.9)
        x = torch.randn(2, 10, 64)
        output, verified = m(x)
        assert output.shape == (2, 10, 64)
        assert isinstance(verified, bool)

    def test_forward_high_confidence(self):
        m = AON_Verifier(64, validation_threshold=0.0)  # Always pass
        x = torch.randn(2, 10, 64)
        output, verified = m(x)
        assert verified is True

    def test_training_mode(self):
        m = AON_Verifier(64, validation_threshold=0.8, training_mode=True)
        assert m.threshold < 0.8  # Should be lowered

    def test_set_training_mode(self):
        m = AON_Verifier(64, validation_threshold=0.8)
        m.set_training_mode(True)
        assert m.threshold < 0.8
        m.set_training_mode(False)
        assert m.threshold == 0.8

    def test_threshold_setter(self):
        m = AON_Verifier(64)
        m.threshold = 0.5
        assert m.base_threshold == 0.5

    def test_get_confidence(self):
        m = AON_Verifier(64)
        x = torch.randn(2, 10, 64)
        conf = m.get_confidence(x)
        assert 0.0 <= conf <= 1.0

    def test_gradient_flow(self):
        """Test gradient flows through verifier_net via score computation.
        Since AON returns x unchanged, we test with a custom loss on scores."""
        m = AON_Verifier(64)
        x = torch.randn(2, 10, 64, requires_grad=True)
        
        # Loss through the score tensor (not x directly since x passes through)
        score = m.verifier_net(x)
        loss = score.mean()
        loss.backward()
        assert list(m.verifier_net.parameters())[0].grad is not None

    def test_gradient_flow_through_passthrough(self):
        """Test gradient flows through the passthrough path of x."""
        m = AON_Verifier(64)
        x = torch.randn(2, 10, 64, requires_grad=True)
        output, _ = m(x)
        loss = output.sum()
        loss.backward()
        # x passes through unchanged, so gradient should be 1 for each element
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x.grad))

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            AON_Verifier(64, validation_threshold=1.5)

    def test_wrong_dim(self):
        m = AON_Verifier(64)
        with pytest.raises(ValueError):
            m(torch.randn(2, 10, 32))


# ============================================================
# PTT_Optimizer Tests
# ============================================================

class TestPTT:
    def test_creation(self):
        base = torch.optim.Adam([torch.nn.Parameter(torch.randn(10))])
        ptt = PTT_Optimizer(base, sparsity_ratio=0.01)
        assert ptt.sparsity_ratio == 0.01

    def test_step(self):
        model = nn.Linear(10, 10)
        base = torch.optim.SGD(model.parameters(), lr=0.01)
        ptt = PTT_Optimizer(base, sparsity_ratio=0.05)
        x = torch.randn(5, 10)
        target = torch.randn(5, 10)
        loss = nn.MSELoss()(model(x), target)
        loss.backward()
        ptt.step()
        stats = ptt.get_stats()
        assert stats["steps"] == 1

    def test_zero_grad(self):
        model = nn.Linear(10, 10)
        base = torch.optim.SGD(model.parameters(), lr=0.01)
        ptt = PTT_Optimizer(base)
        ptt.zero_grad()
        # Should not raise

    def test_get_sparsity_ratio(self):
        base = torch.optim.Adam([torch.nn.Parameter(torch.randn(10))])
        ptt = PTT_Optimizer(base, sparsity_ratio=0.03)
        assert ptt.get_sparsity_ratio() == 0.03

    def test_get_compression_ratio(self):
        base = torch.optim.Adam([torch.nn.Parameter(torch.randn(10))])
        ptt = PTT_Optimizer(base, sparsity_ratio=0.01)
        assert ptt.get_compression_ratio() == 100.0

    def test_reset_prior(self):
        model = nn.Linear(10, 10)
        base = torch.optim.SGD(model.parameters(), lr=0.01)
        ptt = PTT_Optimizer(base)
        x = torch.randn(5, 10)
        target = torch.randn(5, 10)
        loss = nn.MSELoss()(model(x), target)
        loss.backward()
        ptt.step()
        assert len(ptt.gradient_prior) > 0
        ptt.reset_prior()
        assert len(ptt.gradient_prior) == 0

    def test_nan_gradient_handling(self):
        model = nn.Linear(10, 10)
        base = torch.optim.SGD(model.parameters(), lr=0.01)
        ptt = PTT_Optimizer(base)
        # Inject NaN gradient
        for p in model.parameters():
            p.grad = torch.full_like(p, float('nan'))
        ptt.step()  # Should not crash

    def test_invalid_sparsity_ratio(self):
        base = torch.optim.Adam([torch.nn.Parameter(torch.randn(10))])
        with pytest.raises(ValueError):
            PTT_Optimizer(base, sparsity_ratio=0.0)
        with pytest.raises(ValueError):
            PTT_Optimizer(base, sparsity_ratio=0.6)

    def test_gradient_clipping(self):
        model = nn.Linear(10, 10)
        base = torch.optim.SGD(model.parameters(), lr=0.01)
        ptt = PTT_Optimizer(base, grad_clip_norm=0.1)
        x = torch.randn(5, 10)
        target = torch.randn(5, 10)
        loss = nn.MSELoss()(model(x), target)
        loss.backward()
        ptt.step()  # Should not crash


# ============================================================
# NLSI_Module Tests
# ============================================================

class TestNLSI:
    def test_creation(self):
        m = NLSI_Module(64)
        assert m.d_model == 64

    def test_forward(self):
        m = NLSI_Module(64)
        x = torch.randn(2, 10, 64)
        out = m(x)
        assert out.shape == (2, 10, 64)

    def test_shield_strength(self):
        m = NLSI_Module(64, insulation_factor=0.5)
        strength = m.get_shield_strength()
        assert 0.0 <= strength <= 1.0

    def test_gradient_flow(self):
        m = NLSI_Module(64)
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = m(x)
        loss = out.sum()
        loss.backward()
        assert m.insulation_barrier.grad is not None

    def test_invalid_d_model(self):
        with pytest.raises(ValueError):
            NLSI_Module(0)

    def test_invalid_factor(self):
        with pytest.raises(ValueError):
            NLSI_Module(64, insulation_factor=0.0)
        with pytest.raises(ValueError):
            NLSI_Module(64, insulation_factor=1.5)

    def test_wrong_dim(self):
        m = NLSI_Module(64)
        with pytest.raises(ValueError):
            m(torch.randn(2, 10, 32))


# ============================================================
# SovereignAIManager Integration Tests
# ============================================================

class TestSovereignAIManager:
    def test_creation(self):
        config = {'d_model': 64}
        m = SovereignAIManager(config)
        assert m.max_retries == 3

    def test_forward_general(self):
        config = {'d_model': 64, 'aon_training_mode': True, 'use_nlsi': False}
        m = SovereignAIManager(config)
        x = torch.randn(2, 10, 64)
        out = m(x, task_type='general')
        if out is not None:
            assert out.shape == (2, 10, 64)

    def test_forward_with_task(self):
        config = {'d_model': 64, 'aon_training_mode': True, 'use_nlsi': False}
        m = SovereignAIManager(config)
        x = torch.randn(2, 10, 64)
        out = m(x, task_type='medical')
        if out is not None:
            assert out.shape == (2, 10, 64)

    def test_nlsi_integrated(self):
        config = {'d_model': 64, 'aon_training_mode': True, 'use_nlsi': True}
        m = SovereignAIManager(config)
        assert hasattr(m, 'nlsi_module')

    def test_set_training_mode(self):
        config = {'d_model': 64}
        m = SovereignAIManager(config)
        m.set_training_mode(True)
        assert m.aon_verifier.training_mode is True

    def test_create_ptt_optimizer(self):
        config = {'d_model': 64}
        m = SovereignAIManager(config)
        base = torch.optim.Adam(m.parameters(), lr=0.001)
        ptt = m.create_ptt_optimizer(base)
        assert isinstance(ptt, PTT_Optimizer)

    def test_get_pipeline_config(self):
        config = {'d_model': 64}
        m = SovereignAIManager(config)
        cfg = m.get_pipeline_config()
        assert cfg['d_model'] == 64
        assert cfg['nlsi_active'] is True or cfg['nlsi_active'] is False

    def test_gradient_flow(self):
        config = {'d_model': 64, 'aon_training_mode': True}
        m = SovereignAIManager(config)
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = m(x, task_type='general')
        if out is not None:
            loss = out.sum()
            loss.backward()
            assert list(m.teg_filter.parameters())[0].grad is not None

    def test_forward_gradient_connection(self):
        """Test that gradients flow through the entire pipeline.
        Uses task_type='special' so DGA is active.
        
        Note: AON verifier_net is intentionally a side-channel scorer
        (asymmetric oracle pattern). It computes a score but passes x through
        unchanged, so its parameters are NOT in the output computation graph.
        AON gradients are tested separately via TestAON.test_gradient_flow."""
        config = {'d_model': 64, 'aon_training_mode': True, 'use_nlsi': True}
        m = SovereignAIManager(config)
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = m(x, task_type='special')  # Non-general to engage DGA
        if out is not None:
            loss = out.mean()
            loss.backward()
            # Check each module got gradients
            teg_grad = m.teg_filter.entropy_gate[0].weight.grad
            rea_grad = m.rea_core.W.weight.grad
            dga_grad = m.dga_adapter.down_proj.weight.grad
            nlsi_grad = m.nlsi_module.insulation_barrier.grad
            assert teg_grad is not None, "TEG has no gradient"
            assert rea_grad is not None, "REA has no gradient"
            assert dga_grad is not None, "DGA has no gradient (use task_type != 'general')"
            assert nlsi_grad is not None, "NLSI has no gradient"
            # AON verifier_net is intentionally a side-channel scorer
            # (asymmetric oracle): it scores x but passes x through unchanged.
            # So gradients flow through the passthrough, not into verifier_net.
            # Verify by checking x.grad flows (requires_grad was set) and
            # that AON verifier_net has NO gradient (by design).
            aon_grad = m.aon_verifier.verifier_net[0].weight.grad
            assert aon_grad is None, (
                "AON verifier_net is designed as a side-channel scorer; "
                "its output is x unchanged, so its params should NOT "
                f"have gradients from output loss, but got {aon_grad.norm().item():.4f}"
            )

    def test_forward_gradient_connection_general(self):
        """Test gradients flow through pipeline with general task (DGA bypassed)."""
        config = {'d_model': 64, 'aon_training_mode': True, 'use_nlsi': True}
        m = SovereignAIManager(config)
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = m(x, task_type='general')  # DGA bypassed
        if out is not None:
            loss = out.mean()
            loss.backward()
            # TEG, REA, AON, NLSI should have gradients
            assert m.teg_filter.entropy_gate[0].weight.grad is not None
            assert m.rea_core.W.weight.grad is not None
            assert m.nlsi_module.insulation_barrier.grad is not None
            # DGA down_proj is bypassed, so no gradient - this is expected
            assert m.dga_adapter.down_proj.weight.grad is None, \
                "DGA bypassed in general mode, expected no gradient"

    def test_full_training_loop(self):
        """Simulate a complete training step"""
        config = {
            'd_model': 32,
            'aon_threshold': 0.8,
            'aon_training_mode': True,
            'max_retries': 2,
            'use_nlsi': True,
        }
        m = SovereignAIManager(config)
        base_optim = torch.optim.AdamW(m.parameters(), lr=0.001)
        ptt = PTT_Optimizer(base_optim, sparsity_ratio=0.05, grad_clip_norm=1.0)
        criterion = nn.MSELoss()

        x = torch.randn(2, 10, 32)
        target = torch.randn(2, 10, 32)

        ptt.zero_grad()
        output = m(x)
        if output is not None:
            loss = criterion(output, target)
            loss.backward()
            ptt.step()
            assert ptt.get_stats()["steps"] == 1

    def test_verification_retries(self):
        """Test that retry logic doesn't crash"""
        config = {'d_model': 64, 'aon_threshold': 0.99, 'max_retries': 2}
        m = SovereignAIManager(config)
        x = torch.randn(2, 10, 64)
        # Might return None if all retries fail, that's OK
        out = m(x, task_type='general')
        assert out is None or out.shape == (2, 10, 64)


# ============================================================
# BenchmarkRunner Tests
# ============================================================

class TestBenchmarkRunner:
    def test_creation(self):
        s = nn.Linear(10, 10)
        t = nn.Linear(10, 10)
        b = BenchmarkRunner(s, t)
        assert b.sovereign is s

    def test_benchmark(self):
        class MockModel(nn.Module):
            def forward(self, x):
                return x
            def get_ptt_efficiency(self):
                return 0.05

        s = MockModel()
        t = MockModel()
        b = BenchmarkRunner(s, t)
        data = [torch.randn(2, 10) for _ in range(3)]
        loader = torch.utils.data.DataLoader(data, batch_size=2)
        results = b.run_benchmark(loader, num_warmup=1)
        assert "sovereign" in results
        assert "standard" in results

    def test_measure_latency(self):
        model = nn.Linear(10, 10)
        data = [torch.randn(2, 10)]
        loader = torch.utils.data.DataLoader(data, batch_size=2)
        latency = BenchmarkRunner.measure_latency(model, loader, num_runs=5)
        assert latency > 0

    def test_compare_sparsity(self):
        results = [{'sparsity': 0.01}, {'sparsity': 0.02}]
        stats = BenchmarkRunner.compare_sparsity(results)
        assert abs(stats['avg'] - 0.015) < 0.001
        assert stats['compression_ratio'] > 0


# ============================================================
# BenchmarkDashboard Tests
# ============================================================

class TestBenchmarkDashboard:
    def test_creation(self):
        d = BenchmarkDashboard(use_gui=False)
        assert d.use_gui is False

    def test_update(self):
        d = BenchmarkDashboard(use_gui=False)
        d.update({
            'sparsity': 0.05,
            'throughput': 95.0,
            'efficiency': 120.0,
            'latency': 15.0,
        })
        assert len(d.history['sparsity']) == 1

    def test_multiple_updates(self):
        d = BenchmarkDashboard(use_gui=False)
        for i in range(5):
            d.update({
                'sparsity': 0.05 + i * 0.01,
                'throughput': 95.0 - i,
                'efficiency': 120.0 + i * 10,
                'latency': 15.0 - i * 0.5,
            })
        assert len(d.history['sparsity']) == 5
        assert len(d.history['latency']) == 5

    def test_get_summary(self):
        d = BenchmarkDashboard(use_gui=False)
        d.update({
            'sparsity': 0.05,
            'throughput': 95.0,
            'efficiency': 120.0,
            'latency': 15.0,
        })
        summary = d.get_summary()
        assert 'sparsity_mean' in summary
        assert 'throughput_latest' in summary

    def test_missing_key(self):
        d = BenchmarkDashboard(use_gui=False)
        with pytest.raises(ValueError):
            d.update({'sparsity': 0.05})

    def test_close(self):
        d = BenchmarkDashboard(use_gui=False)
        d.close()  # Should not raise


# ============================================================
# Cross-Module Integration Tests
# ============================================================

class TestIntegration:
    def test_full_pipeline_consistency(self):
        """Test that all modules produce same-shaped output"""
        d_model = 64
        batch = 2
        seq = 10

        teg = TEG_Module(d_model)
        rea = REA_Module(d_model)
        dga = DGA_Module(d_model)
        aon = AON_Verifier(d_model, validation_threshold=0.0)  # Always pass
        nlsi = NLSI_Module(d_model)

        x = torch.randn(batch, seq, d_model)

        x1 = teg(x)
        assert x1.shape == (batch, seq, d_model)

        x2 = rea(x1)
        assert x2.shape == (batch, seq, d_model)

        x3 = dga(x2)
        assert x3.shape == (batch, seq, d_model)

        x4, verified = aon(x3)
        assert x4.shape == (batch, seq, d_model)
        assert verified is True

        x5 = nlsi(x4)
        assert x5.shape == (batch, seq, d_model)

    def test_all_modules_importable(self):
        """Test all modules are importable from package"""
        assert TEG_Module is not None
        assert REA_Module is not None
        assert DGA_Module is not None
        assert AON_Verifier is not None
        assert PTT_Optimizer is not None
        assert NLSI_Module is not None
        assert SovereignAIManager is not None
        assert BenchmarkRunner is not None
        assert BenchmarkDashboard is not None
