"""Compatibility import path for the RISC-V instruction coverage extension."""

from analysis.rv_instruction_coverage import (
    BATCH_SCHEMA,
    EXPERIMENT_COVERAGE_PROFILE,
    EXPERIMENT_PRIVILEGE_MODE,
    METRICS,
    RV_INSTRUCTION_METRICS,
    SCHEMA,
    aggregate_rv_instruction_coverage,
    aggregate_summary,
    attach_case_metrics,
    attach_rv_instruction_metrics,
    case_metrics,
    compare_experiment_unions,
    registry_for_profile,
)

__all__ = [
    "BATCH_SCHEMA",
    "EXPERIMENT_COVERAGE_PROFILE",
    "EXPERIMENT_PRIVILEGE_MODE",
    "METRICS",
    "RV_INSTRUCTION_METRICS",
    "SCHEMA",
    "aggregate_summary",
    "aggregate_rv_instruction_coverage",
    "attach_case_metrics",
    "attach_rv_instruction_metrics",
    "case_metrics",
    "compare_experiment_unions",
    "registry_for_profile",
]
