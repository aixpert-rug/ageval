import time
from unittest.mock import patch

import pytest

from metrics.efficiency.cost_per_query import CostPerQueryEvaluator
from metrics.efficiency.energy_carbon import EnergyCarbonEvaluator
from metrics.efficiency.latency import LatencyEvaluator
from metrics.efficiency.memory_footprint import MemoryFootprintEvaluator
from metrics.efficiency.throughput import ThroughputEvaluator


def test_latency_measure_and_evaluate_real_timing_sanity_check():
    """Loose-tolerance real-wall-clock test: proves live timing genuinely
    works end to end, not just that mocking is set up correctly."""
    def slow_work():
        time.sleep(0.05)
        return "done"

    result_value, eval_result = LatencyEvaluator().measure_and_evaluate(slow_work, budget_seconds=1.0)
    assert result_value == "done"
    assert 0.04 < eval_result.metadata["elapsed_seconds"] < 0.5


def test_latency_measure_and_evaluate_precise_via_mocked_clock():
    with patch("time.perf_counter", side_effect=[0.0, 2.5]):
        result_value, eval_result = LatencyEvaluator().measure_and_evaluate(lambda: "x", budget_seconds=10.0)
    assert eval_result.metadata["elapsed_seconds"] == 2.5
    assert eval_result.metadata["band_level"] == 4


def test_memory_measure_and_evaluate_default_tracemalloc_measurer():
    def allocate():
        _ = [0] * 1_000_000
        return "allocated"

    result_value, eval_result = MemoryFootprintEvaluator().measure_and_evaluate(allocate, budget_mb=1000.0)
    assert result_value == "allocated"
    assert eval_result.metadata["memory_mb"] > 0


def test_memory_measure_and_evaluate_custom_measurer():
    """Confirms the measurer is genuinely pluggable, not hardcoded --
    the concrete use case being a future GPU-aware measurer."""
    def fake_gpu_measurer(fn):
        result = fn()
        return result, 8000.0

    result_value, eval_result = MemoryFootprintEvaluator().measure_and_evaluate(
        lambda: "gpu work", budget_mb=16000.0, memory_measurer=fake_gpu_measurer
    )
    assert result_value == "gpu work"
    assert eval_result.metadata["memory_mb"] == 8000.0
    # ratio 8000/16000 = 0.5, exactly at the band-3 edge (<=0.50) --
    # verified directly against contracts.banding.ratio_to_budget_band
    # before writing this assertion.
    assert eval_result.metadata["band_level"] == 3


def test_throughput_measure_and_evaluate_precise_via_mocked_clock():
    with patch("time.perf_counter", side_effect=[0.0, 2.0]):
        work_units, eval_result = ThroughputEvaluator().measure_and_evaluate(lambda: 200, budget=100.0)
    assert work_units == 200
    assert eval_result.metadata["throughput"] == 100.0
    assert eval_result.metadata["band_level"] == 4


def test_throughput_rejects_near_zero_elapsed_time():
    with patch("time.perf_counter", side_effect=[0.0, 1e-9]):
        with pytest.raises(ValueError, match="measurement floor"):
            ThroughputEvaluator().measure_and_evaluate(lambda: 100, budget=50.0)


def test_cost_compute_and_evaluate_exact_calculation():
    result = CostPerQueryEvaluator().compute_and_evaluate(
        input_tokens=1000, output_tokens=500,
        price_per_1k_input=0.001, price_per_1k_output=0.002,
        budget=0.01,
    )
    assert result.metadata["cost"] == pytest.approx(0.002)
    assert result.metadata["band_level"] == 4


def test_cost_compute_and_evaluate_rejects_negative_tokens():
    with pytest.raises(ValueError):
        CostPerQueryEvaluator().compute_and_evaluate(
            input_tokens=-1, output_tokens=500, price_per_1k_input=0.001, price_per_1k_output=0.002, budget=0.01
        )


def test_energy_estimate_and_evaluate_exact_calculation_via_mocked_clock():
    """1 hour elapsed (mocked), 300W draw, 400 gCO2/kWh grid intensity:
    energy = 300 * 1 / 1000 = 0.3 kWh; carbon = 0.3 * 400 = 120 gCO2e.
    120/200 = 0.6 ratio -> band 2 ("Adequate"), verified directly against
    contracts.banding.ratio_to_budget_band before writing this assertion."""
    with patch("time.perf_counter", side_effect=[0.0, 3600.0]):
        result_value, eval_result = EnergyCarbonEvaluator().estimate_and_evaluate(
            lambda: "work done", avg_power_watts=300.0, grid_carbon_intensity_g_per_kwh=400.0, budget_gco2e=200.0
        )
    assert result_value == "work done"
    assert eval_result.metadata["energy_kwh"] == pytest.approx(0.3)
    assert eval_result.metadata["value"] == pytest.approx(120.0)
    assert eval_result.metadata["is_rough_estimate"] is True
    assert eval_result.metadata["band_level"] == 2


def test_energy_estimate_requires_positive_power():
    with pytest.raises(ValueError):
        EnergyCarbonEvaluator().estimate_and_evaluate(
            lambda: None, avg_power_watts=0.0, grid_carbon_intensity_g_per_kwh=400.0, budget_gco2e=200.0
        )


def test_energy_estimate_rejects_negative_carbon_intensity():
    with pytest.raises(ValueError):
        EnergyCarbonEvaluator().estimate_and_evaluate(
            lambda: None, avg_power_watts=300.0, grid_carbon_intensity_g_per_kwh=-1.0, budget_gco2e=200.0
        )


def test_energy_measure_with_codecarbon_real_measurement():
    """Real, not mocked: codecarbon actually runs and measures."""
    def light_work():
        time.sleep(0.3)
        return "done"

    result_value, eval_result = EnergyCarbonEvaluator().measure_with_codecarbon_and_evaluate(
        light_work, budget_gco2e=1000.0
    )
    assert result_value == "done"
    assert eval_result.metadata["measurement_source"] == "codecarbon"
    assert eval_result.metadata["energy_kwh"] >= 0.0
    assert eval_result.metadata["duration_seconds"] > 0.0
    assert eval_result.metadata["value"] >= 0.0


def test_energy_measure_with_codecarbon_water_tracking_when_wue_provided():
    """Confirms water tracking activates when wue_l_per_kwh is passed --
    codecarbon itself defaults water_consumed to 0.0 unless told a WUE."""
    def light_work():
        time.sleep(0.2)
        return "done"

    _, eval_result = EnergyCarbonEvaluator().measure_with_codecarbon_and_evaluate(
        light_work, budget_gco2e=1000.0, wue_l_per_kwh=1.8
    )
    assert "water_liters" in eval_result.metadata
    assert eval_result.metadata["water_liters"] >= 0.0


def test_energy_estimate_and_evaluate_still_available_as_fallback():
    with patch("time.perf_counter", side_effect=[0.0, 3600.0]):
        _, eval_result = EnergyCarbonEvaluator().estimate_and_evaluate(
            lambda: "x", avg_power_watts=300.0, grid_carbon_intensity_g_per_kwh=400.0, budget_gco2e=200.0
        )
    assert eval_result.metadata["measurement_source"] == "rough_estimator"
    assert eval_result.metadata["is_rough_estimate"] is True
