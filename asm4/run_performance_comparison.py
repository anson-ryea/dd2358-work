#!/usr/bin/env python3
"""
Task 1.3: Performance comparison — Serial vs Multiprocessing vs Dask.
Run outside Jupyter so Dask can use process-based workers (real parallelism).

Usage (from asm4/ directory). Use the same Python as your Jupyter kernel (e.g. conda env):
  conda activate profiling   # or your env name
  cd asm4 && python run_performance_comparison.py

Requires: numpy, dask, distributed. Screenshot the terminal output for the notebook.
"""
import sys
import os

# Run from asm4 directory so wildfire_sim_worker is found
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir != os.getcwd():
    os.chdir(_script_dir)
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

try:
    import time
    import numpy as np
    from multiprocessing import Pool, cpu_count
    from wildfire_sim_worker import run_one_simulation, DAYS
except ImportError as e:
    print("Import error:", e)
    print("Install dependencies in the same Python you use for Jupyter:")
    print("  pip install numpy dask distributed")
    print("Then run from terminal:  cd asm4  &&  python run_performance_comparison.py")
    sys.exit(1)

N_RUNS = 8
n_repeats = 2


def run_serial():
    results = [run_one_simulation(i) for i in range(N_RUNS)]
    return np.array([r + [0] * (DAYS - len(r)) for r in results], dtype=float).mean(axis=0)


def run_multiprocessing():
    n_workers = min(N_RUNS, cpu_count())
    with Pool(processes=n_workers) as pool:
        results = pool.map(run_one_simulation, range(N_RUNS))
    padded = [np.array(r + [0] * (DAYS - len(r)), dtype=float) for r in results]
    return np.mean(padded, axis=0)


def run_dask(n_workers=4, use_processes=True):
    try:
        from dask import delayed
        import dask.array as da
        from dask.distributed import Client, LocalCluster
    except ImportError:
        raise ImportError("Install dask and distributed: pip install dask distributed")

    def padded(_):
        from wildfire_sim_worker import run_one_simulation, DAYS
        import numpy as np
        r = run_one_simulation(_)
        return np.array(r + [0] * (DAYS - len(r)), dtype=np.float64)

    cluster = LocalCluster(processes=use_processes, n_workers=n_workers)
    client = Client(cluster)
    delayed_sims = [delayed(padded)(i) for i in range(N_RUNS)]
    dask_arrays = [da.from_delayed(d, shape=(DAYS,), dtype=np.float64) for d in delayed_sims]
    stacked = da.stack(dask_arrays)
    result = da.mean(stacked, axis=0).compute(scheduler=client)
    client.close()
    return result


def main():
    print("=" * 60)
    print("Task 1.3: Performance comparison (run outside Jupyter)")
    print("N_RUNS =", N_RUNS, "| n_repeats =", n_repeats)
    print("=" * 60)

    times = {}
    for name, fn in [
        ("Serial", run_serial),
        ("Multiprocessing", run_multiprocessing),
        ("Dask (4 workers)", lambda: run_dask(4)),
    ]:
        t_list = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            fn()
            t_list.append(time.perf_counter() - t0)
        times[name] = min(t_list)
        print(f"{name}: {times[name]:.2f} s (min of {n_repeats} runs)")

    print("\n--> Fastest:", min(times, key=times.get))
    print("\nDone. Screenshot this output for the notebook.")


if __name__ == "__main__":
    main()
