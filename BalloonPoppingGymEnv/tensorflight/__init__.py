"""Public TensorFlight training API.

TensorFlight is a competition-specific, batched PyTorch environment.  The
official ActiveRocketPy environment remains the evaluation oracle; policies
trained here must be validated there before they are treated as candidates.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "PPOHyperparameters",
    "RolloutMetrics",
    "TensorAgentObservation",
    "TensorEnvironmentStep",
    "TensorFlightEnvironment",
    "TensorFlightEnvironmentConfig",
    "TensorFlightSource",
    "TensorFlightTrainer",
    "TensorFlightTrainingConfig",
    "TrainingUpdateMetrics",
    "build_scenario1_source",
    "make_tensorflight_environment",
]

_EXPORTS = {
    "PPOHyperparameters": ("ppo", "PPOHyperparameters"),
    "RolloutMetrics": ("trainer", "RolloutMetrics"),
    "TensorAgentObservation": ("environment", "TensorAgentObservation"),
    "TensorEnvironmentStep": ("environment", "TensorEnvironmentStep"),
    "TensorFlightEnvironment": ("environment", "TensorFlightEnvironment"),
    "TensorFlightEnvironmentConfig": ("environment", "TensorFlightEnvironmentConfig"),
    "TensorFlightSource": ("factory", "TensorFlightSource"),
    "TensorFlightTrainer": ("trainer", "TensorFlightTrainer"),
    "TensorFlightTrainingConfig": ("trainer", "TensorFlightTrainingConfig"),
    "TrainingUpdateMetrics": ("trainer", "TrainingUpdateMetrics"),
    "build_scenario1_source": ("factory", "build_scenario1_source"),
    "make_tensorflight_environment": ("factory", "make_tensorflight_environment"),
}


def __getattr__(name: str):
    """Import PyTorch only when a TensorFlight runtime symbol is requested."""

    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    try:
        module = import_module(f"{__name__}.{module_name}")
    except ModuleNotFoundError as error:
        if error.name == "torch":
            raise ModuleNotFoundError(
                "TensorFlight requires PyTorch; install the tensorflight extra "
                "and a CUDA-enabled PyTorch build when training on GPU"
            ) from error
        raise
    value = getattr(module, attribute)
    globals()[name] = value
    return value
