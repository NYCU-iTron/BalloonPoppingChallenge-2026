# Repository Guidelines

## Project Structure & Module Organization

`BalloonPoppingGymEnv/` is the main Python package. Environment physics and scenario YAML files live under `envs/`; competitor and example implementations belong in `agents/`; evaluation entry points, configs, and result helpers are in `evaluation/`. `tests/` contains the root regression and contract suite, with golden data in `tests/baselines/`. Documentation examples and figures are under `doc/`, while submission utilities live in `scripts/`. `ActiveRocketPy/` is a pinned Git submodule and has its own tests and tooling; do not update it incidentally.

## Build, Test, and Development Commands

Initialize a fresh checkout before running the simulator:

```shell
git submodule update --init --recursive --checkout
uv sync --locked --extra dev
```

- `uv run python BalloonPoppingGymEnv/evaluation/evaluate.py BalloonPoppingGymEnv/evaluation/configs/example_eval_cfg.yaml` runs the example evaluation.
- `uv run --no-sync pytest tests/ --cov=BalloonPoppingGymEnv` runs the fast suite with coverage.
- `BPC_RUN_SLOW_TESTS=1 uv run --no-sync pytest tests/` includes the scenario-1 Monte Carlo regression used by CI.
- `make format` fixes import order and formats package, test, and example code.
- `uv build` creates the wheel and source distribution through Hatchling.

## Coding Style & Naming Conventions

Use four-space indentation and standard Python naming: `snake_case` for modules, functions, and variables; `PascalCase` for classes; leading underscores for private helpers. Ruff 0.15.20 is authoritative. Before review, run:

```shell
uvx ruff@0.15.20 check BalloonPoppingGymEnv/ tests/ doc/examples/ scripts/
uvx ruff@0.15.20 format --check BalloonPoppingGymEnv/ tests/ doc/examples/ scripts/
```

Keep optional `vpython` imports lazy. Name scenario files consistently, for example `scenario_1_parameters.yaml`.

## Testing Guidelines

Add focused tests as `tests/test_<behavior>.py`; test methods use `test_<expected_behavior>`. Both pytest-style tests and `unittest.TestCase` classes are supported. Cover behavior changes and failure paths. Physics changes require deliberate baseline regeneration with `PYTHONPATH=. python tests/baselines/regenerate_scenario_0.py` (and scenario 1), followed by review of numeric and score diffs. No fixed coverage percentage is specified, but CI publishes coverage.

## Commit & Pull Request Guidelines

Follow the history’s short, sentence-case, imperative summaries, such as `Record the seed a run used`. Branch from and target `develop`; `main` is the release line. PRs must explain what changed, why, and how it was checked; link issues (`Closes #123`), select a release-note label, and add user-visible changes to `CHANGELOG.md` under `Unreleased`. Include screenshots for rendering or visualization changes.

## Security & Configuration

Do not commit credentials or private submissions. Report exploitable issues privately through GitHub’s Security tab, as described in `SECURITY.md`. Competitors should limit changes to `BalloonPoppingGymEnv/agents/`; evaluation integrity checks treat simulator and evaluation code as fixed.
