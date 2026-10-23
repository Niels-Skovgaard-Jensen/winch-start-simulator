"""Simulation jobs for the GUI, run in a separate process so they can be stopped.

A diffrax solve is one long compiled call that a thread cannot interrupt; a
process can simply be terminated.  This module does not import Qt, so the worker
process starts quickly.

Compilation: the worker compiles the model once at start-up (a warm-up launch with
the default settings), in the background.  JAX's persistent compilation cache is
switched on too, but it only skips the XLA compilation step (about 2 of the ~16 s);
most of the start-up time is JAX tracing the Python code, which has to be redone
in every new process (e.g. after a stop).
"""

import dataclasses
import multiprocessing as mp
import queue
import time
import traceback
from pathlib import Path

import jax
import numpy as np

import winch_sim  # noqa: F401  (float64)

from ..cable import ROPES
from ..gliders import CATALOGUE
from ..params import Env, Launch
from ..sensitivity import sensitivities
from ..simulate import (
    EVENTS,
    LAUNCH_END_EVENTS,
    T_MAX,
    n_segments_for,
    solve_launch,
    summarize,
    time_series,
)
from ..winch import engine_winch

CACHE_DIR = Path.home() / ".cache" / "winch_sim" / "jax"


@dataclasses.dataclass
class Numerics:
    segments_per_km: float = 10.0
    t_max: float = T_MAX
    rtol: float = 1e-6
    atol: float = 1e-6
    t_post: float = 8.0
    sensitivity: bool = True
    sensitivity_tol: float = 1e-8


@dataclasses.dataclass
class RunResult:
    name: str
    launch: Launch
    numerics: Numerics
    n_segments: int
    summary: dict
    series: dict
    sens: tuple | None  # (height, rows)
    note: str
    elapsed: float


def default_launch(glider="ASK 21", rope="dyneema") -> Launch:
    g, p = CATALOGUE[glider]
    return Launch(
        glider=g,
        rope=ROPES[rope],
        winch=engine_winch(g),
        pilot=p,
        env=Env(),
        rope_length=1200.0,
        slack=0.0,
    )


def to_numpy(launch: Launch) -> Launch:
    """Plain numpy leaves: picklable across processes, independent of devices."""
    return jax.tree.map(np.asarray, launch)


def run_launch(name: str, launch: Launch, num: Numerics) -> RunResult:
    """Simulate one launch (+ sensitivities)."""
    t0 = time.perf_counter()
    n = n_segments_for(launch.rope_length, num.segments_per_km)
    sol = solve_launch(
        launch,
        n_segments=n,
        t_max=num.t_max,
        t_post=num.t_post,
        rtol=num.rtol,
        atol=num.atol,
    )
    summary = summarize(launch, sol)
    series = time_series(launch, sol)
    sens, note = None, ""
    if num.sensitivity:
        if summary["event"] in EVENTS[:LAUNCH_END_EVENTS]:
            sens = sensitivities(
                launch, n_segments=n, tol=num.sensitivity_tol, t_max=num.t_max
            )
        else:
            note = f"no sensitivities: launch ended with {summary['event']}"
    return RunResult(
        name,
        to_numpy(launch),
        num,
        n,
        summary,
        series,
        sens,
        note,
        time.perf_counter() - t0,
    )


WARMUP = "warmup"


def worker_main(requests, results, warmup: bool = True) -> None:
    """Worker process: run jobs from `requests`, put results on `results`.

    With `warmup`, first run one launch with the default settings. That compiles
    the simulation and the gradient for the default rope discretisation and
    solver settings in the background, and reports (WARMUP, status, seconds).
    Later runs with the same array shapes and solver settings reuse the compiled
    code; only a different number of rope segments or other solver settings
    trigger a new compilation.
    """
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(CACHE_DIR))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
    except OSError:
        pass  # no cache: slower after a stop, but correct
    if warmup:
        t0 = time.perf_counter()
        try:
            run_launch(WARMUP, to_numpy(default_launch()), Numerics())
            results.put((WARMUP, "ok", time.perf_counter() - t0))
        except Exception:  # noqa: BLE001
            results.put((WARMUP, "error", traceback.format_exc()))
    while True:
        job = requests.get()
        if job is None:
            return
        job_id, name, launch, num = job
        try:
            results.put((job_id, "ok", run_launch(name, launch, num)))
        except Exception:  # noqa: BLE001  (report anything to the GUI)
            results.put((job_id, "error", traceback.format_exc()))


class SimProcess:
    """One worker process; `stop()` kills it (and any running job) at once."""

    def __init__(self, warmup: bool = True):
        self._ctx = mp.get_context("spawn")  # fork is unsafe with Qt/JAX threads
        self._warmup = warmup
        self._proc = None
        self._requests = None
        self._results = None

    def start(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            return
        self._requests = self._ctx.Queue()
        self._results = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=worker_main,
            args=(self._requests, self._results, self._warmup),
            daemon=True,
            name="winch-sim-worker",
        )
        self._proc.start()

    def submit(self, job_id: int, name: str, launch: Launch, num: Numerics) -> None:
        self.start()
        assert self._requests is not None
        self._requests.put((job_id, name, to_numpy(launch), num))

    def poll(self):
        """(job_id, "ok"|"error", payload) if a job finished, "dead" if the worker
        died unexpectedly, else None."""
        if self._results is None:
            return None
        try:
            return self._results.get_nowait()
        except queue.Empty:
            if self._proc is not None and not self._proc.is_alive():
                return "dead"
            return None

    def stop(self) -> None:
        """Kill the worker (and whatever it is computing)."""
        if self._proc is not None:
            self._proc.terminate()
            self._proc.join(2.0)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(2.0)
        for q in (self._requests, self._results):
            if q is not None:
                q.cancel_join_thread()
                q.close()
        self._proc = self._requests = self._results = None

    def shutdown(self) -> None:
        self.stop()
