# AGENTS.md

Rules for coding agents that change this repo. Facts about the project are in the "For AI agents" section of [README.md](README.md).

## Where to run

- Edit and commit on the MacBook. Run tests and experiments on `ssh sandbox`, in `~/p/zyx1121/wifi-wall-mapping/main/`.
- Claim the sandbox directory first: `claim zyx1121/wifi-wall-mapping <session>`. Release it when you finish.
- Copy the code to the sandbox with rsync. Do not copy `.git`, `.venv` or `results/`:

  ```sh
  rsync -a --delete --exclude .git --exclude .venv --exclude results/ ./ sandbox:p/zyx1121/wifi-wall-mapping/main/
  ```

- Run Python with `uv run`. Do not install packages with pip.

## Traps

- Set BLAS to 1 thread. On the sandbox VM, a multithreaded OpenBLAS `eigh` on a 320 x 320 matrix takes 26 s. With 1 thread it takes 0.03 s. `experiments/common.py` and `tests/conftest.py` set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` before numpy loads. Keep `import common` as the first import in every experiment script.
- Experiments run in parallel processes (`pool_map`). Do not add threads inside one trial.
- Each stage uses its own seeds: stage 2 uses 1000+, stage 3 uses 2000+, stage 4 reports 5000+. Seeds 3000 to 3009 tuned the fusion parameters. Do not report results from tuning seeds.
- The OMP and IFFT estimators are on-grid. Their tests use a 0.3 m and 3 degree tolerance. The other estimators use 0.05 m and 1 degree.

## Conventions

- Geometry is 2D. The AP sits on the south wall at `y = 0.1` with a uniform linear array along x. The array measures only `cos(phi)`, so there is a front/back ambiguity. The mount wall is not a target.
- Wi-Fi gives delay relative to the LoS path, not absolute time of flight. The LoS length `d0` comes from the truth, from FTM, or from `fusion.hough_d0`.
- A wall error is the distance between the estimated line and the midpoint of the true wall, along the estimated normal. A match needs 0.5 m and 10 degrees.
- Results go to `results/<stage>/` as JSON and PNG. Git ignores `results/raw/`.

## Verify

```sh
uv run pytest -q
```

All 8 tests must pass. For an estimator change, also run `experiments/stage2.py 20` and compare the reference row (160 MHz, 4 antennas, 30 dB) against `results/stage2/stage2.json`.
