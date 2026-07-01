__all__ = ["BalloonPoppingRLWrapper", "RLTrainingConfig"]


def __getattr__(name):
    if name in __all__:
        from scripts.rl_training.run_training_framework import (
            BalloonPoppingRLWrapper,
            RLTrainingConfig,
        )

        exports = {
            "BalloonPoppingRLWrapper": BalloonPoppingRLWrapper,
            "RLTrainingConfig": RLTrainingConfig,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
