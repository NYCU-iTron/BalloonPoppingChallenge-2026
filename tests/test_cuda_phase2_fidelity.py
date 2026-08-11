"""Phase 2 gates for the competition-specific tensor rocket model."""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from experiments.cuda.phase2_oracle import (  # noqa: E402
    apply_canonical_actuator_command,
    build_scenario1_oracle,
    canonical_actuator_outputs,
    canonical_components,
    generate_phase2_report,
    scenario1_actuator_specs,
)
from experiments.cuda.phase2_rocket import (  # noqa: E402
    Scenario1ActuatorBank,
    Scenario1TensorRocket,
    dopri5_step,
    rk4_step,
)


@pytest.fixture(scope="module")
def oracle():
    result = build_scenario1_oracle(seed=2041)
    yield result
    result.close()


@pytest.fixture(scope="module")
def report():
    return generate_phase2_report(steps=64, seed=2042)


def test_training_rhs_module_has_no_canonical_or_scipy_imports() -> None:
    path = Path("experiments/cuda/phase2_rocket.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert imported_roots.isdisjoint(
        {"rocketpy", "ActiveRocketPy", "BalloonPoppingGymEnv", "scipy", "numpy"}
    )


def test_scenario1_actuator_filter_rate_limit_and_clamp_match_official(oracle) -> None:
    flight = oracle.flight
    specs, demand_rate = scenario1_actuator_specs(flight)
    bank = Scenario1ActuatorBank(
        1,
        specs,
        demand_rate=demand_rate,
        dtype=torch.float64,
    )
    rocket = flight.rocket
    for actuator in (
        rocket.roll_control,
        rocket.thrust_vector_control.x,
        rocket.thrust_vector_control.y,
        rocket.throttle_control,
    ):
        actuator._reset()

    commands = np.asarray(
        [
            [100.0, 100.0, -100.0, -4.0],
            [-100.0, -100.0, 100.0, 4.0],
            [0.1, 0.2, -0.3, 0.5],
            [20.0, 30.0, 30.0, 0.0],
        ],
        dtype=np.float64,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for command in commands:
            expected = apply_canonical_actuator_command(flight, command)
            actual = np.asarray(bank.update(torch.from_numpy(command[None]))[0])
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    # Rate limiting remains active even though all Scenario 1 time constants are null.
    bank.reset()
    first = np.asarray(bank.update(torch.tensor([[100.0, 100.0, -100.0, -4.0]]))[0])
    np.testing.assert_allclose(first, [0.2, 0.6, -0.6, 0.98], rtol=0.0, atol=1e-15)
    assert all(spec.time_constant is None for spec in specs)


def test_masked_actuator_reset_does_not_alias_input(oracle) -> None:
    specs, demand_rate = scenario1_actuator_specs(oracle.flight)
    bank = Scenario1ActuatorBank(2, specs, demand_rate=demand_rate)
    bank.update(torch.tensor([[10.0, 10.0, 10.0, 0.0]]).expand(2, -1))
    mask = torch.tensor([True, False])
    bank.reset(mask)
    assert mask.tolist() == [True, False]
    torch.testing.assert_close(bank.output[0], bank.initial)
    assert not torch.equal(bank.output[1], bank.initial)


def test_float64_rhs_components_match_canonical_snapshot(oracle) -> None:
    flight = oracle.flight
    model = Scenario1TensorRocket(oracle.model_data, dtype=torch.float64)
    state = np.asarray(flight.y_sol, dtype=np.float64)
    actuator = canonical_actuator_outputs(flight)
    canonical = canonical_components(flight, flight.t, state, actuator)
    actual = model.components(
        flight.t - oracle.launch_time,
        torch.from_numpy(state[None]),
        torch.from_numpy(actuator[None]),
    )
    for name in actual.__dataclass_fields__:
        np.testing.assert_allclose(
            np.asarray(getattr(actual, name)[0]),
            canonical[name],
            rtol=1e-10,
            atol=2e-7,
            err_msg=name,
        )


def test_mass_com_inertia_and_burnout_branch_match_at_fixed_times(oracle) -> None:
    flight = oracle.flight
    model = Scenario1TensorRocket(oracle.model_data, dtype=torch.float64)
    state = np.asarray(flight.y_sol, dtype=np.float64)
    actuator = np.array([0.0, 0.0, 0.0, 1.0])
    elapsed_times = np.asarray((0.00125, 1.3375, 29.99875, 30.00125, 30.5))
    for elapsed in elapsed_times:
        canonical = canonical_components(
            flight,
            oracle.launch_time + elapsed,
            state,
            actuator,
        )
        actual = model.components(
            elapsed,
            torch.from_numpy(state[None]),
            torch.from_numpy(actuator[None]),
        )
        np.testing.assert_allclose(
            actual.total_mass[0], canonical["total_mass"], atol=1e-10
        )
        np.testing.assert_allclose(
            actual.center_of_mass[0], canonical["center_of_mass"], atol=1e-9
        )
        np.testing.assert_allclose(actual.inertia[0], canonical["inertia"], atol=1e-8)
        np.testing.assert_allclose(actual.thrust[0], canonical["thrust"], atol=1e-10)
    assert (
        float(
            model.components(
                29.99875,
                torch.from_numpy(state[None]),
                torch.from_numpy(actuator[None]),
            ).thrust[0, 2]
        )
        > 0
    )
    assert (
        float(
            model.components(
                30.00125,
                torch.from_numpy(state[None]),
                torch.from_numpy(actuator[None]),
            ).thrust[0, 2]
        )
        == 0
    )


def test_generalized_rhs_matches_diverse_burning_and_coast_states(oracle) -> None:
    flight = oracle.flight
    model = Scenario1TensorRocket(oracle.model_data, dtype=torch.float64)
    cases = (
        (
            0.5,
            [
                12.0,
                -7.0,
                100.0,
                5.0,
                -3.0,
                50.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.01,
                -0.02,
                0.03,
            ],
            [0.1, 0.2, -0.3, 0.95],
        ),
        (
            10.0,
            [
                100.0,
                50.0,
                1000.0,
                80.0,
                30.0,
                120.0,
                0.97296034,
                0.10241688,
                0.20483376,
                0.05120844,
                0.2,
                -0.15,
                0.1,
            ],
            [2.0, 3.0, -2.5, 0.7],
        ),
        (
            29.9,
            [
                -300.0,
                150.0,
                3000.0,
                -100.0,
                60.0,
                80.0,
                0.92387953,
                0.0,
                0.38268343,
                0.0,
                -0.3,
                0.25,
                -0.2,
            ],
            [-3.0, -4.0, 3.5, 0.4],
        ),
        (
            30.1,
            [
                400.0,
                -250.0,
                4500.0,
                30.0,
                -90.0,
                -60.0,
                0.8660254,
                0.28867513,
                -0.28867513,
                0.28867513,
                0.4,
                0.1,
                -0.35,
            ],
            [1.0, 5.0, -5.0, 1.0],
        ),
        (
            100.0,
            [
                50.0,
                25.0,
                2000.0,
                -40.0,
                20.0,
                -100.0,
                0.96592583,
                0.25881905,
                0.0,
                0.0,
                -0.1,
                0.05,
                0.2,
            ],
            [0.0, 0.0, 0.0, 0.0],
        ),
    )

    for elapsed, state_values, actuator_values in cases:
        state = np.asarray(state_values, dtype=np.float64)
        state[6:10] /= np.linalg.norm(state[6:10])
        actuator = np.asarray(actuator_values, dtype=np.float64)
        expected = canonical_components(
            flight,
            oracle.launch_time + elapsed,
            state,
            actuator,
        )
        actual = model.components(
            elapsed,
            torch.from_numpy(state[None]),
            torch.from_numpy(actuator[None]),
        )
        for name in actual.__dataclass_fields__:
            tolerance = (
                2e-4
                if name in {"center_of_mass_ddot", "translation_acceleration", "rhs"}
                else 2e-7
            )
            np.testing.assert_allclose(
                np.asarray(getattr(actual, name)[0]),
                expected[name],
                rtol=1e-9,
                atol=tolerance,
                err_msg=f"elapsed={elapsed}, component={name}",
            )


def test_rhs_component_report_passes_float64_gates(report) -> None:
    assert report["dtype"] == "float64"
    assert report["compared_steps"] == 64
    errors = report["rhs_component_errors"]
    strict = (
        "total_mass",
        "total_mass_dot",
        "center_of_mass",
        "inertia",
        "inertia_dot",
        "wind",
        "thrust",
        "aerodynamic_force",
        "aerodynamic_moment",
        "control_moment",
        "angular_acceleration",
        "quaternion_derivative",
    )
    for name in strict:
        assert errors[name]["p99"] <= 1e-8, (name, errors[name])
    assert errors["atmosphere"]["p99"] <= 1e-7
    # The canonical COM second derivative itself is a 1e-6 central difference
    # of a composite Function, so it is the limiting float64 table quantity.
    assert errors["center_of_mass_ddot"]["p99"] <= 2e-4
    assert errors["translation_acceleration"]["p99"] <= 2e-4
    assert errors["rhs"]["p99"] <= 2e-4


def test_rk4_substeps_pass_against_fresh_rhs_reference(report) -> None:
    fresh = report["integrator_errors"]["rk4_vs_fresh_dopri5"]
    for substeps in ("1", "2", "4"):
        errors = fresh[substeps]
        assert errors["position_m"]["p99"] <= 1e-7
        assert errors["velocity_m_s"]["p99"] <= 5e-6
        assert errors["attitude_deg"]["p99"] <= 1e-4
        assert errors["angular_rate_rad_s"]["p99"] <= 5e-6


def test_stale_fsal_compatibility_matches_unmodified_official_step(report) -> None:
    errors = report["integrator_errors"]["dopri5_stale_fsal_vs_official"]
    assert errors["position_m"]["p99"] <= 1e-7
    assert errors["velocity_m_s"]["p99"] <= 1e-5
    assert errors["attitude_deg"]["p99"] <= 1e-4
    assert errors["angular_rate_rad_s"]["p99"] <= 1e-6

    # A fresh-RHS RK4 solve should intentionally differ from the unmodified
    # official step when a new control command invalidates SciPy's FSAL cache.
    fresh_rk4 = report["integrator_errors"]["rk4_vs_official_stale_fsal"]["4"]
    assert fresh_rk4["angular_rate_rad_s"]["p99"] > 1e-5


def test_dopri5_and_rk4_preserve_batch_shape_and_float64(oracle) -> None:
    model = Scenario1TensorRocket(oracle.model_data, dtype=torch.float64)
    state = torch.as_tensor(oracle.flight.y_sol, dtype=torch.float64).expand(3, -1)
    actuator = torch.zeros((3, 4), dtype=torch.float64)
    actuator[:, 3] = 1.0
    rk4 = rk4_step(model, 0.02, state, actuator, substeps=4)
    dopri, endpoint = dopri5_step(model, 0.02, state, actuator)
    assert rk4.shape == dopri.shape == endpoint.shape == (3, 13)
    assert rk4.dtype == dopri.dtype == endpoint.dtype == torch.float64
    assert torch.isfinite(rk4).all()
    assert torch.isfinite(dopri).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_phase2_rhs_and_integrators_stay_on_cuda(oracle) -> None:
    model = Scenario1TensorRocket(
        oracle.model_data,
        device="cuda",
        dtype=torch.float64,
    )
    state = torch.as_tensor(
        oracle.flight.y_sol,
        device="cuda",
        dtype=torch.float64,
    ).expand(32, -1)
    actuator = torch.zeros((32, 4), device="cuda", dtype=torch.float64)
    actuator[:, 3] = 1.0

    derivative = model.rhs(0.02, state, actuator)
    rk4 = rk4_step(model, 0.02, state, actuator, substeps=1)
    dopri, endpoint = dopri5_step(model, 0.02, state, actuator)

    assert derivative.device.type == "cuda"
    assert rk4.device.type == dopri.device.type == endpoint.device.type == "cuda"
    assert derivative.dtype == rk4.dtype == dopri.dtype == torch.float64
