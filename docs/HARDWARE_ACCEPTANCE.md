# Hardware acceptance and low-resource configuration

As requested, the available host was tested under real Linux limits because no
separate GPU target or older PC is available. These are actual resource controls,
not a claim that throttling a modern processor reproduces an old processor.
No driver, numerical backend or system service was installed for these tests.

## Completed low-resource gate — 2026-11-23

Host: Ryzen3 7320U, about7GiB kernel-visible RAM, Linux7.3.5, Python3.12.16.
Each profile ran in its own fresh transient user service with CPUQuota100%,
MemoryMax1GiB, MemorySwapMax0 and TasksMax64. Backend thread counts were one.
Three fresh subprocess samples per case used isolated Python and installed NIL,
with no SDK source path injected. Existing small-profile budgets were unchanged.

| Profile | Cases | Kernel unit peak | Maximum measured case wall time | Result |
| --- | --- | --- | --- | --- |
| Core | Import and native regression | 30.17MiB | 0.198s | Passed |
| Classical | Core, native regression, direct sklearn and NIL Ridge | 121.12MiB | 1.179s | Passed |
| Neural | Core, native regression and small Torch training | 239.35MiB | 2.940s | Passed |

All cases passed numerical correctness/learning and declared time/RSS budgets;
all three units recorded zero OOM kills and zero swap use. Classical NIL fit
added about2.96ms over the independent direct-backend sample medians, including
explicit run/artifact recording. Small timings include noise; this is not a
speedup claim. The Ridge workload has2048x8 features; neural training has512x8
inputs, width16 and two epochs. This does not promise large models fit in1GiB.

Kernel unit peak is the cgroup charge, including its controller and workers.
Process RSS also includes shared resident mappings and is recorded separately;
these two memory measurements are not interchangeable. CPUQuota100% limits CPU
time per scheduling period; it does not emulate an older instruction set, clock,
cache, storage or thermal behavior. Filesystem cache was not forcibly dropped.

Raw reports include all samples, numerical results, versions and kernel counters:
[core](benchmarks/2026-11-23-low-resource-core.json),
[classical](benchmarks/2026-11-23-low-resource-classical.json),
[neural](benchmarks/2026-11-23-low-resource-neural.json).
The original old-PC gate stays open and old_pc_acceptance remains false.

## Reproduce on an explicitly provisioned Linux machine

Install the regular NIL wheel and only the optional backend needed for the chosen
profile. Use an absolute Python executable inside that environment. OpenSSL is
not needed for these workloads. A working user systemd manager/cgroup-v2 is needed.
Adapt `/absolute/venv/bin/python`, repository and output paths below:

```sh
systemd-run --user --wait --pipe --collect \
  --unit=nil-low-resource-core \
  -p MemoryMax=1G -p MemorySwapMax=0 -p CPUQuota=100% -p TasksMax=64 \
  --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 \
  --setenv=MKL_NUM_THREADS=1 \
  /absolute/venv/bin/python /absolute/NIL/scripts/accept_low_resource.py \
  --profile core --repeats 3 --output /absolute/results/core.json
```

Repeat with distinct unit names, `--profile classical` in the classical/tabular
environment, and `--profile neural` in the CPU Torch environment. Completed units
are collected. The runner reads and verifies real kernel policy before executing;
missing/looser limits fail with exit2. Workload or budget failure gives exit1 and
retains diagnostics. It never changes controllers/ancestors or upgrades packages.

For an actual older PC, record CPU model, installed RAM and approximate machine
age/provenance, then run the same workloads/budgets on it. Keep its reports
separate from the current-host constrained results. A user-supplied host label
alone is not evidence of machine age or instruction compatibility.

## GPU status and remaining gates

The actual [GPU preflight](benchmarks/2026-11-23-gpu-preflight.json) failed with
exit2: Radeon610M display adapter (AMD1002:1506, amdgpu), Torch2.15.0+cpu,
no CUDA or HIP runtime, no CUDA devices. The repository had no configured
self-hosted GPU runners when checked. No GPU training tests executed, and no
permanent workflow was queued against an unavailable runner.

A display adapter is not proof that the installed numerical backend can use it.
This acceptance runner targets CUDA; it does not certify AMD/Intel GPU training.
Do not install a CUDA wheel on this AMD-only host and count that as GPU support.
The configured [manual GPU workflow](../.github/workflows/sdk-gpu.yml) requires
an explicitly provisioned trusted Linux runner labelled nil-cuda with matching
CUDA-enabled Torch, driver metadata, pytest and build tooling. It installs NIL
without replacing the operator's backend. It requires one GPU for AMP and two
visible CUDA GPUs plus NCCL for DDP/FSDP. Choose only a precision actually supported
by the target; a requested unsupported dtype fails instead of passing via skips.

```sh
python scripts/accept_gpu.py --mode amp --precision fp16 --output gpu-fp16.json
python scripts/accept_gpu.py --mode all --precision all --output gpu-all.json
```

Real CUDA gates remain: FP16/BF16 training and exact supported resume, overflow/
unscale/clipping, two-GPU DDP/FSDP parity and fault recovery, sharded memory
measurements, and device-specific hard aggregate VRAM containment. Allocator
preferences do not implement hard VRAM quotas. These need actual suitable targets.
