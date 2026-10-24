"""Integration checks: the target registry, hardware routing on Modal (fakes, no network),
the per-target images, the smoke CLI, the LM prepare hook, dev mock-latency wiring, and the
"modal only in workers/runners/modal_*.py" rule (D28)."""

from __future__ import annotations

import ast
import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from backend.benchmark.configs import RaceConfig
from backend.core.types import ArmKind
from workers.runners.base import RunnerError
from workers.targets.base import TARGET_MODULES, get_target, list_targets, load_all_targets

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_HARDWARE = {"lm": "L4", "cifar": "L4", "rl": "CPU"}


# --------------------------------------------------------------------------- registry


def test_registry_has_the_three_real_targets() -> None:
    assert set(TARGET_MODULES) == set(EXPECTED_HARDWARE) | {"lcbench"}  # lcbench: surrogate (SurrogateRunner)
    reg = load_all_targets()
    assert set(EXPECTED_HARDWARE) <= set(reg)
    assert set(EXPECTED_HARDWARE) <= set(list_targets())  # other tests register fakes
    for name, hw in EXPECTED_HARDWARE.items():
        t = get_target(name)
        assert t.spec.name == name and t.name == name
        assert t.spec.hardware == hw
        assert not t.spec.is_trap(t.spec.default_config())
        t.spec.validate_config(t.spec.default_config())
    with pytest.raises(KeyError):
        get_target("toy")


def test_importing_all_targets_is_light() -> None:
    """The backend imports every target spec; that must not pull in torch, jax or modal."""
    code = (
        "import sys; from workers.targets.base import load_all_targets; load_all_targets(); "
        "bad = [m for m in ('torch', 'jax', 'optax', 'modal') if m in sys.modules]; "
        "print(','.join(bad)); sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"heavy imports: {r.stdout} {r.stderr[-500:]}"


def test_modal_imported_only_in_modal_runner() -> None:
    """D28: ``modal`` may be imported only in ``workers/runners/modal_*.py``."""
    runners = ROOT / "workers" / "runners"
    allowed = set(runners.glob("modal_*.py"))
    assert runners / "modal_runner.py" in allowed
    offenders: list[str] = []
    for top in ("backend", "workers", "scripts"):
        for path in (ROOT / top).rglob("*.py"):
            if path in allowed:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names = [node.module]
                if any(n == "modal" or n.startswith("modal.") for n in names):
                    offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


# --------------------------------------------------------------------------- Modal routing

modal_runner = pytest.importorskip("workers.runners.modal_runner")
ModalRunner = modal_runner.ModalRunner


class _Aio:
    def __init__(self, fn: Any) -> None:
        self.aio = fn


class _StubQueue:
    def __init__(self) -> None:
        self.get_many = _Aio(self._get_many)
        self.clear = _Aio(self._clear)

    async def _get_many(self, n: int, block: bool = True, timeout: float | None = None,
                        partition: str | None = None) -> list[Any]:
        import queue as q

        if block:
            await asyncio.sleep(timeout or 0)
            raise q.Empty
        return []

    async def _clear(self, partition: str | None = None, **_: Any) -> None:
        return None


class _StubCall:
    def __init__(self, n: int) -> None:
        self.object_id = f"fc-stub{n}"
        self.get = _Aio(self._get)
        self.cancel = _Aio(self._cancel)
        self.cancelled = False

    async def _get(self, timeout: float | None = None) -> Any:
        await asyncio.sleep(min(timeout or 0, 0.01))
        if self.cancelled:  # like Modal: get() raises once the input is cancelled
            raise RuntimeError("InputCancellation (stub)")
        raise TimeoutError

    async def _cancel(self, terminate_containers: bool = False) -> None:
        self.cancelled = True


class _StubFunction:
    def __init__(self, name: str) -> None:
        self.name = name
        self.jobs: list[dict[str, Any]] = []
        self.spawn = _Aio(self._spawn)

    async def _spawn(self, job: dict[str, Any]) -> _StubCall:
        self.jobs.append(job)
        return _StubCall(len(self.jobs))


@pytest.fixture
def stub_modal():
    ModalRunner._global_busy.clear()
    fns: dict[str, _StubFunction] = {}

    def resolver(name: str) -> _StubFunction:
        return fns.setdefault(name, _StubFunction(name))

    yield fns, resolver
    ModalRunner._global_busy.clear()


async def test_modal_routes_rl_to_cpu_and_torch_targets_to_l4(stub_modal) -> None:
    fns, resolver = stub_modal
    runner = ModalRunner({"L4": 2, "CPU": 1}, function_resolver=resolver, queue=_StubQueue(), poll_interval_s=0.02)
    handles = []
    for name in ("rl", "lm", "cifar"):
        spec = get_target(name).spec
        handles.append(await runner.launch(name, spec.default_config(), 0, spec.budget_s, lambda p: None))
    assert [j["target"] for j in fns["run_cpu"].jobs] == ["rl"]
    assert [j["target"] for j in fns["run_l4"].jobs] == ["lm", "cifar"]
    assert {h.target: h.hardware for h in handles} == EXPECTED_HARDWARE
    assert runner.busy_slots("CPU") == 1 and runner.busy_slots("L4") == 2
    for h in handles:
        await runner.cancel(h)
        r = await runner.wait(h)
        assert r.status.value == "cancelled"
    assert runner.busy_slots() == 0
    # CPU price is per core times the reserved cores.
    assert runner.price_per_second("CPU") == pytest.approx(0.0000131 * runner.cpu_cores)


async def test_modal_refuses_target_without_matching_slots(stub_modal) -> None:
    _, resolver = stub_modal
    l4_only = ModalRunner({"L4": 1}, function_resolver=resolver, queue=_StubQueue())
    spec = get_target("rl").spec
    with pytest.raises(RunnerError):
        await l4_only.launch("rl", spec.default_config(), 0, spec.budget_s, lambda p: None)


def test_modal_images_match_targets() -> None:
    gpu = " ".join(modal_runner.IMAGE_PIP_PACKAGES)
    cpu = " ".join(modal_runner.CPU_IMAGE_PIP_PACKAGES)
    assert "torch" in gpu and "jax" not in gpu
    assert "jax==" in cpu and "optax" in cpu and "torch" not in cpu and "gymnax" not in cpu
    assert modal_runner.IMAGE_ENV["LM_DATA_DIR"].startswith(modal_runner.DATA_MOUNT)
    assert modal_runner.IMAGE_ENV["BUSY_DATA_DIR"] == modal_runner.DATA_MOUNT


def test_smoke_cli_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[Any, ...]] = []

    async def fake_smoke(budget_s: Any = None, config_overrides: Any = None, seed: int = 0,
                         target: str = "cifar") -> dict[str, Any]:
        seen.append((target, config_overrides, budget_s))
        return {}

    monkeypatch.setattr(modal_runner, "smoke", fake_smoke)
    modal_runner._smoke_cli([])
    modal_runner._smoke_cli(["rl"])
    modal_runner._smoke_cli(["lm", '{"n_layer": 2}', "20"])
    modal_runner._smoke_cli(['{"epochs": 8}', "45"])
    assert seen == [("cifar", {}, None), ("rl", {}, None), ("lm", {"n_layer": 2}, 20), ("cifar", {"epochs": 8}, 45)]


# --------------------------------------------------------------------------- targets


def test_lm_prepare_hook_uses_existing_corpus(tmp_path: Path) -> None:
    from workers.targets import lm

    (tmp_path / lm.CORPUS_FILENAME).write_bytes(b"hello world\n" * 10)
    path = get_target("lm").prepare(str(tmp_path))  # type: ignore[attr-defined]
    assert Path(path) == tmp_path / lm.CORPUS_FILENAME
    assert hasattr(get_target("cifar"), "prepare")


# --------------------------------------------------------------------------- dev wiring


def test_mock_latency_per_arm_reaches_mock_llm() -> None:
    from backend.app import default_arm_deps

    cfg = RaceConfig(target="rl", arms=[ArmKind.opus, ArmKind.mubit], budget_s=10, gpus_per_arm=1,
                     mock_latency_s={ArmKind.opus: 8.0, ArmKind.mubit: 0.3}, mock_trap_rate=0.25)
    spec = get_target("rl").spec
    opus = default_arm_deps("r", cfg, spec, ArmKind.opus)
    mubit = default_arm_deps("r", cfg, spec, ArmKind.mubit)
    assert opus.researcher.llm.latency_s == 8.0  # type: ignore[attr-defined]
    assert mubit.researcher.llm.latency_s == 0.3  # type: ignore[attr-defined]
    assert mubit.researcher.llm.trap_rate == 0.25  # type: ignore[attr-defined]
    assert opus.llm_mode == "mock" and opus.runner.name == "local"


def test_mock_latency_rejected_outside_mock_mode() -> None:
    with pytest.raises(ValueError):
        RaceConfig(target="rl", budget_s=10, gpus_per_arm=1, llm="mubit", mock_latency_s={ArmKind.opus: 1.0})
    with pytest.raises(ValueError):
        RaceConfig(target="rl", budget_s=10, gpus_per_arm=1, mock_latency_s={ArmKind.opus: -1.0})


# --------------------------------------------------------------------------- scripts


def test_bench_preflight_refuses_unfrozen_presets() -> None:
    import importlib.util

    spec_ = importlib.util.spec_from_file_location("bench_script", ROOT / "scripts" / "bench.py")
    assert spec_ is not None and spec_.loader is not None
    bench = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(bench)
    from backend.benchmark.configs import PRESETS

    for name, cfg in PRESETS.items():
        res = {n: ok for n, ok, _ in bench.checks(cfg, {})}  # {} = no goal file
        assert res["GPU $ estimate within guard"], name  # every preset fits its own $ guard
        assert res["no H100"], name
        assert not res["target_value frozen"], name      # nothing frozen without the goal file
    missing = ROOT / "data" / "calibration" / "does-not-exist.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "bench.py"), "run", "demo-3m", "--goals", str(missing)],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "refused" in r.stderr
