"""Updated server-two queue, preserving the historical execution semantics."""

import os
from pathlib import Path

ROOT = Path(os.environ.get("DAG_OUTPUT_ROOT", str(Path(__file__).resolve().parent.parent / "outputs/runtime")))
ASSETS = ROOT.parent
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "JULIA_NUM_THREADS"):
    os.environ[key] = "1"
for key, folder in (("XDG_CONFIG_HOME", "config328_v1"), ("XDG_CACHE_HOME", "cache328_v1"),
                    ("XDG_DATA_HOME", "data_config328_v1"), ("NUMBA_CACHE_DIR", "numba_cache328_v1"),
                    ("MPLCONFIGDIR", "mplconfig328_v1")):
    os.environ[key] = str(ASSETS / folder)
os.environ.update(PYTHONNOUSERSITE="1", CASTLE_BACKEND="pytorch", WANDB_MODE="disabled")

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timedelta, timezone
import contextlib
import json
import multiprocessing
import socket
import time
import traceback

import psutil


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def physical_cores():
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path("/sys/devices/system/cpu/cpu%d/topology" % cpu)
        key = tuple((topology / field).read_text().strip() for field in ("physical_package_id", "core_id"))
        cores.setdefault(key, cpu)
    return list(cores.values())


def bind(cpus):
    index = multiprocessing.current_process()._identity[0] - 1
    os.sched_setaffinity(0, {cpus[index % len(cpus)]})


def operate(function, task, stage):
    began = time.monotonic()
    identifier = task["identifier"]
    path = ROOT / "task_receipts" / stage / (identifier + ".json")
    if path.exists() and read(path).get("status") == "completed":
        return read(path)
    log = path.with_suffix(".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open("a") as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            result = function(task)
    except Exception:
        result = dict(status="ERROR", traceback=traceback.format_exc())
    result.update(identifier=identifier, stage=stage, seconds=time.monotonic() - began,
                  worker_pid=os.getpid(), affinity=sorted(os.sched_getaffinity(0)), beijing=stamp())
    if path.exists():
        attempt = 1
        previous = path.with_name(path.stem + "_attempt%d.json" % attempt)
        while previous.exists():
            attempt += 1
            previous = path.with_name(path.stem + "_attempt%d.json" % attempt)
        path.rename(previous)
    save(path, result)
    return result


def execute(stage, function, tasks, maximum=64, start_method="fork"):
    if os.environ.get("DAG_REQUIRE_HOST") and socket.gethostname() != os.environ["DAG_REQUIRE_HOST"]:
        raise RuntimeError("This execution requires the configured compute host")
    available = psutil.virtual_memory().available / 2 ** 30
    cpus = physical_cores()[:maximum]
    workers = min(len(tasks), len(cpus), int(max(0., available - 128.) // 4))
    if workers < 1:
        raise RuntimeError("The 128 GiB MemAvailable reserve or task allocation cannot be met")
    save(ROOT / (stage + "_launch.json"), dict(beijing=stamp(), pid=os.getpid(), tasks=len(tasks),
         workers=workers, cpus=cpus[:workers], MemAvailable_GiB=available, reserve_GiB=128,
         monitor_interval_seconds=10800, errors_continue=True))
    records = []
    next_check = time.monotonic() + 10800
    with ProcessPoolExecutor(max_workers=workers, initializer=bind, initargs=(cpus[:workers],),
                             mp_context=multiprocessing.get_context(start_method)) as pool:
        pending = {pool.submit(operate, function, task, stage): task for task in tasks}
        processes = list(pool._processes.values())
        save(ROOT / (stage + "_startup.json"), dict(beijing=stamp(), worker_pids=[p.pid for p in processes]))
        while pending:
            done, _ = wait(pending, timeout=max(.1, min(60., next_check - time.monotonic())), return_when=FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                try:
                    record = future.result()
                except Exception:
                    record = dict(status="ERROR", identifier=task["identifier"], traceback=traceback.format_exc())
                    save(ROOT / "task_receipts" / stage / (task["identifier"] + ".json"), record)
                records.append(record)
                if len(records) % 64 == 0 or len(records) == len(tasks) or record["status"] != "completed":
                    print(json.dumps(dict(stage=stage, completed=len(records), expected=len(tasks),
                                          status=record["status"], identifier=record["identifier"])), flush=True)
            if time.monotonic() >= next_check:
                resident = sum(psutil.Process(p.pid).memory_info().rss for p in processes if p.is_alive()) / 2 ** 30
                save(ROOT / (stage + "_monitor_temporary.json"), dict(beijing=stamp(), remaining=len(pending),
                     MemAvailable_GiB=psutil.virtual_memory().available / 2 ** 30, resident_GiB=resident))
                next_check += 10800
    report = dict(status="completed" if all(r["status"] == "completed" for r in records) else "errors_to_repair",
                  beijing=stamp(), expected=len(tasks), records=records,
                  distinct_workers=len({r.get("worker_pid") for r in records}),
                  MemAvailable_GiB=psutil.virtual_memory().available / 2 ** 30)
    save(ROOT / (stage + "_completion.json"), report)
    (ROOT / (stage + "_monitor_temporary.json")).unlink(missing_ok=True)
    return report
