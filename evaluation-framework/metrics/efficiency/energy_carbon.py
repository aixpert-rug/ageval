"""
Energy / carbon per query -- Level 2 metric under Efficiency .

Applicability: efficiency.

Real measurement backend: wraps `codecarbon`'s EmissionsTracker --
confirmed working end-to-end during development of this file (real
CPU/RAM power draw measured, real emissions and energy figures
returned, not just imported). This is also the specific instrumentation
tool named by Vector Institute's own "Data and Impact Accounting" (DIA)
position paper (Raza et al., ICML 2026 Position Paper Track --
arXiv:2601.21632, VectorInstitute/ai-impact-accounting), which proposes
CodeCarbon (alongside ML CO2 Impact and cloud provider APIs) as the
recommended low-friction instrumentation layer underneath a standardised
impact-reporting schema.

Note on scope: DIA's own primary framing is CUMULATIVE, MODEL-CARD-LEVEL
footprint tracking across training runs and derivative models (fine-tunes,
LoRAs, quantizations) -- a different scope from this metric, which is
PER-QUERY INFERENCE footprint. The two are related, not identical: this evaluator adopts DIA's
recommended instrumentation tool and reports DIA-schema-aligned fields
(energy_kwh, water_liters, hardware info) so results COULD feed into a
DIA-style dashboard, but doesn't implement DIA's cumulative/lineage
tracking itself.

Water tracking requires an explicit `wue_l_per_kwh` (Water Usage
Effectiveness, litres per kWh) -- codecarbon itself defaults this to 0.0
(no water tracking) unless told otherwise, since WUE is
datacenter/hardware-specific and codecarbon has no way to guess it
correctly. Same "no silent defaults for environment-specific constants"
principle already applied to the fallback estimator below.

Fallback: `estimate_and_evaluate()` (rough elapsed-time x power-draw
estimator) is kept for environments where installing codecarbon isn't
practical -- treat it as clearly lower-fidelity than the codecarbon path.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from contracts.banding import ratio_to_budget_band
from contracts.evaluator import EvaluationResult, WP3Evaluator


class EnergyCarbonEvaluator(WP3Evaluator):
    dimensions = ("Efficiency",)
    applicability = "efficiency"
    metric_id = "energy_carbon"

    def evaluate_measurement(self, value: float, budget: float, unit: str = "gCO2e") -> EvaluationResult:
        if value < 0:
            raise ValueError(f"value must be non-negative, got {value}.")

        band = ratio_to_budget_band(value, budget, lower_is_better=True)
        score = band.level / 4.0

        return EvaluationResult(
            score=score,
            explanation=f"{value:.4f} {unit} against a {budget:.4f} {unit} budget -- band {band.level} ({band.label}).",
            is_success=band.level >= 2,
            metadata={"value": value, "budget": budget, "unit": unit, "band_level": band.level, "band_label": band.label},
        )

    def measure_with_codecarbon_and_evaluate(
        self,
        fn: Callable[[], Any],
        budget_gco2e: float,
        wue_l_per_kwh: float | None = None,
        tracker_kwargs: dict | None = None,
    ) -> tuple[Any, EvaluationResult]:
        """
        Real measurement via codecarbon's EmissionsTracker.

        Args:
            fn: zero-argument callable representing the work to measure.
            budget_gco2e: declared carbon budget.
            wue_l_per_kwh: Water Usage Effectiveness for your
                datacenter/hardware, litres per kWh. Omit to skip water
                tracking (codecarbon's own default) -- pass it explicitly
                once you have a real value for your environment (see
                module docstring for why no default is guessed here).
            tracker_kwargs: extra kwargs forwarded to EmissionsTracker
                (e.g. `force_carbon_intensity_g_co2e_kwh` to override
                codecarbon's auto-detected grid intensity, or `gpu_ids`
                to scope GPU tracking).
        """
        try:
            from codecarbon import EmissionsTracker
        except ImportError as e:
            raise RuntimeError(
                "measure_with_codecarbon_and_evaluate requires the 'codecarbon' "
                "package (pip install codecarbon). Use estimate_and_evaluate() "
                "instead for a rough, dependency-free fallback."
            ) from e

        kwargs = dict(tracker_kwargs or {})
        kwargs.setdefault("output_methods", [])  # disable file/API/logger output by default; non-deprecated API (save_to_* is deprecated as of codecarbon 3.x)
        kwargs.setdefault("log_level", "error")
        if wue_l_per_kwh is not None:
            kwargs["wue"] = wue_l_per_kwh

        tracker = EmissionsTracker(**kwargs)
        tracker.start()
        try:
            result = fn()
        finally:
            tracker.stop()

        data = tracker.final_emissions_data
        carbon_gco2e = data.emissions * 1000  # codecarbon reports kg CO2eq; convert to g for this metric's unit

        eval_result = self.evaluate_measurement(value=carbon_gco2e, budget=budget_gco2e, unit="gCO2e")
        eval_result.metadata.update({
            "energy_kwh": data.energy_consumed,
            "water_liters": data.water_consumed,
            "duration_seconds": data.duration,
            "cpu_power_w": data.cpu_power,
            "gpu_power_w": data.gpu_power,
            "ram_power_w": data.ram_power,
            "country": data.country_name,
            "measurement_source": "codecarbon",
        })
        return result, eval_result

    def estimate_and_evaluate(
        self,
        fn: Callable[[], Any],
        avg_power_watts: float,
        grid_carbon_intensity_g_per_kwh: float,
        budget_gco2e: float,
    ) -> tuple[Any, EvaluationResult]:
        """
        Rough fallback estimator for when codecarbon isn't available.
        energy_kWh = power_watts * elapsed_hours / 1000; carbon_gCO2e =
        energy_kWh * grid_carbon_intensity. No water tracking -- this
        fallback doesn't attempt it at all, unlike the codecarbon path.

        Args:
            avg_power_watts, grid_carbon_intensity_g_per_kwh: REQUIRED,
                no defaults -- any default here would be a specific,
                debatable guess presented with false precision.
        """
        if avg_power_watts <= 0:
            raise ValueError(f"avg_power_watts must be positive, got {avg_power_watts}.")
        if grid_carbon_intensity_g_per_kwh < 0:
            raise ValueError(f"grid_carbon_intensity_g_per_kwh must be non-negative, got {grid_carbon_intensity_g_per_kwh}.")

        start = time.perf_counter()
        result = fn()
        elapsed_hours = (time.perf_counter() - start) / 3600

        energy_kwh = avg_power_watts * elapsed_hours / 1000
        carbon_gco2e = energy_kwh * grid_carbon_intensity_g_per_kwh

        eval_result = self.evaluate_measurement(value=carbon_gco2e, budget=budget_gco2e, unit="gCO2e")
        eval_result.metadata["energy_kwh"] = energy_kwh
        eval_result.metadata["elapsed_hours"] = elapsed_hours
        eval_result.metadata["is_rough_estimate"] = True
        eval_result.metadata["measurement_source"] = "rough_estimator"
        return result, eval_result

    def evaluate(self, input):  # pragma: no cover
        raise NotImplementedError(
            "EnergyCarbonEvaluator operates on a pre-measured value or a live "
            "measurement. Call evaluate_measurement(...), "
            "measure_with_codecarbon_and_evaluate(...), or estimate_and_evaluate(...) instead."
        )
