"""Benchmark the non-canonical tensor flight surrogate on CPU and CUDA.

Examples::

    python -m experiments.cuda.benchmark_tensor_flight
    python -m experiments.cuda.benchmark_tensor_flight --device cuda --compile

The reported unit is batched environment transitions per second, not RK4 calls
per second.  Each transition contains four evaluations of the prototype 6-DoF
and actuator derivatives plus hit/done/reward processing.  This is not a
RocketPy benchmark and says nothing by itself about RocketPy numerical parity.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import time
from dataclasses import dataclass

import torch

from experiments.cuda.tensor_flight import TensorFlightBatch, TensorFlightConfig


DEFAULT_BATCH_SIZES = (1, 20, 64, 256, 1024, 4096)


@dataclass(frozen=True)
class BenchmarkResult:
    device: str
    batch_size: int
    mode: str
    steps: int
    repeats: int
    elapsed_seconds: float
    transitions_per_second: float
    minimum_transitions_per_second: float
    maximum_transitions_per_second: float


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_one(
    *,
    device: torch.device,
    batch_size: int,
    steps: int,
    warmup: int,
    repeats: int,
    use_compile: bool,
) -> BenchmarkResult:
    # Keep episodes alive for the complete timing window; reset and random
    # action generation are intentionally outside the measured region.
    config = TensorFlightConfig(max_steps=warmup + steps + 10)
    environment = TensorFlightBatch(batch_size, device=device, config=config)
    action_generator = torch.Generator(device="cpu")
    action_generator.manual_seed(2026 + batch_size)
    action_bank = torch.empty(
        (32, batch_size, environment.action_size),
        device="cpu",
        dtype=environment.dtype,
    ).uniform_(-1.0, 1.0, generator=action_generator)
    action_bank = action_bank.to(device)
    # Keep mean thrust near hover/flight instead of immediately driving every
    # trajectory into the ground.  Values are still normalized actions.
    action_bank[:, :, 3].clamp_(min=-0.35)

    mode = "eager"
    if use_compile:
        try:
            environment.enable_compile()
            # torch.compile errors commonly surface on first execution.
            environment.step(action_bank[0])
            _synchronize(device)
            mode = "compile"
        except Exception as error:  # pragma: no cover - backend-specific
            environment.disable_compile()
            mode = f"eager (compile fallback: {type(error).__name__})"

    elapsed_samples = []
    for repeat in range(repeats):
        environment.reset(seed=2026 + repeat)
        for index in range(warmup):
            environment.step(action_bank[index % len(action_bank)])
        _synchronize(device)

        start = time.perf_counter()
        for index in range(steps):
            environment.step(action_bank[index % len(action_bank)])
        _synchronize(device)
        elapsed_samples.append(time.perf_counter() - start)

    transitions = batch_size * steps
    rates = [transitions / elapsed for elapsed in elapsed_samples]
    return BenchmarkResult(
        device=str(device),
        batch_size=batch_size,
        mode=mode,
        steps=steps,
        repeats=repeats,
        elapsed_seconds=statistics.median(elapsed_samples),
        transitions_per_second=statistics.median(rates),
        minimum_transitions_per_second=min(rates),
        maximum_transitions_per_second=max(rates),
    )


def _requested_devices(selection: str) -> list[torch.device]:
    if selection == "cpu":
        return [torch.device("cpu")]
    if selection == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable")
        return [torch.device("cuda")]
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("all", "cpu", "cuda"),
        default="all",
        help="benchmark all available devices by default",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="try torch.compile; safely fall back to eager execution",
    )
    args = parser.parse_args()
    if args.steps <= 0 or args.warmup < 0 or args.repeats <= 0:
        parser.error(
            "--steps and --repeats must be positive; --warmup cannot be negative"
        )
    if any(batch_size <= 0 for batch_size in args.batch_sizes):
        parser.error("every batch size must be positive")
    return args


def main() -> None:
    args = _parse_args()
    devices = _requested_devices(args.device)
    print("NON-CANONICAL PROTOTYPE -- NOT ROCKETPY PARITY")
    print(f"Python platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}; GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA: unavailable")
    print()
    print(
        f"{'device':<8} {'batch':>7} {'mode':<42} "
        f"{'seconds':>10} {'median trans/s':>16} {'min..max trans/s':>25}"
    )
    print("-" * 118)
    for device in devices:
        for batch_size in args.batch_sizes:
            result = _run_one(
                device=device,
                batch_size=batch_size,
                steps=args.steps,
                warmup=args.warmup,
                repeats=args.repeats,
                use_compile=args.compile,
            )
            print(
                f"{result.device:<8} {result.batch_size:>7,d} "
                f"{result.mode:<42} {result.elapsed_seconds:>10.4f} "
                f"{result.transitions_per_second:>16,.0f} "
                f"{result.minimum_transitions_per_second:>11,.0f}.."
                f"{result.maximum_transitions_per_second:<11,.0f}"
            )


if __name__ == "__main__":
    main()
