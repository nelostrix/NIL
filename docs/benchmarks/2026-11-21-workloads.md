# Measured current-host NIL workloads

Hardware: AMD Ryzen 3 7320U, Linux x86_64, eight logical CPUs. This is a modern
CPU, not a ten-year-old PC. No old-PC, Windows, macOS or CUDA acceptance is implied.
All raw samples, Python/backend versions, timestamps and hardware details are
in `2026-11-21-classical.json` and `2026-11-21-neural.json` beside this report.

Each workload ran in three fresh child interpreters, with one BLAS/Torch thread.
Filesystem caches were not flushed, machine load was not controlled, and there
is no statistical claim of a small performance difference from three samples.
Peak RSS is the worker's measured Linux high-water mark, not a quota.

| Workload | Median fresh-process wall time | Max peak RSS |
|---|---:|---:|
| Core NIL import, classical interpreter | 0.074 s | 19.9 MiB |
| Native sklearn Ridge fit + predict | 1.305 s | 186.0 MiB |
| NIL Ridge fit + native predict | 1.306 s | 186.1 MiB |
| Core NIL import, neural interpreter | 0.080 s | 20.2 MiB |
| NIL tiny Torch CPU training | 3.414 s | 305.9 MiB |

Core import loads none of NumPy, sklearn or Torch. It starts no process/thread.
The core-import process wall time includes interpreter and benchmark startup;
see the samples' `workload_s` for the separately timed import interval.

Ridge: 100,000 rows, 24 float64 features, seeded synthetic linear target, one
fit and prediction over all rows; NIL also creates its real run metadata in a
temporary directory. Median fit-only time was 0.022 s native and 0.028 s NIL;
median prediction time was approximately 0.003 s in both. Native numerical
inference is used directly. Process startup/backend imports dominate this case.
This is a dense linear workload, not representative of every classical model.

Neural: 8,192 rows, 24 float32 features, sum target, 24→64→1 ReLU MLP, two epochs,
batch size 128, default NIL optimizer/checkpoint publication, zero loader workers.
Median training interval was 1.290 s; final sample losses are in the JSON.
Optional Torch dominates memory compared with the dependency-free NIL import.

Reproduce from the repository with the appropriate optional environment:

```sh
PYTHONPATH=nil-sdk python scripts/benchmark_workloads.py --profile classical --repeats 3 --output classical.json
PYTHONPATH=nil-sdk python scripts/benchmark_workloads.py --profile neural --repeats 3 --output neural.json
```

These measurements do not certify low-end hardware or large-scale training.
The checklist benchmark gate remains open until the same harness runs on
actual documented older hardware with agreed latency/RAM targets. Large sparse,
image/sequence, distributed, long-duration and peak aggregate execution workloads
also require separate evidence.
