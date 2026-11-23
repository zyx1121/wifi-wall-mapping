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
- Walks: we tuned `fuse_walk` and `close_rooms` on apartment A (`floorplan.apartment`, walk seed 0). Apartment B (`floorplan.apartment_b`) with walk seeds 7000+ is the held-out set. Report plan B. Do not tune any threshold on it. The wall-loss threshold in `attenuation.false_lines` also comes from plan A.
- The OMP and IFFT estimators are on-grid. Their tests use a 0.3 m and 3 degree tolerance. The other estimators use 0.05 m and 1 degree.

## Conventions

- Geometry is 2D. The AP sits on the south wall at `y = 0.1` with a uniform linear array along x. The array measures only `cos(phi)`, so there is a front/back ambiguity. The mount wall is not a target.
- Wi-Fi gives delay relative to the LoS path, not absolute time of flight. The LoS length `d0` comes from the truth, from FTM, from `fusion.hough_d0`, or from the EKF track of `wallmap/pdr.py`.
- A wall error is the distance between the estimated line and the midpoint of the true wall, along the estimated normal. A match needs 0.5 m and 10 degrees.
- Results go to `results/<stage>/` as JSON and PNG. Git ignores `results/raw/`. Stages 5 and 6 cache their links there: remove the cache after you change the channel or an estimator.
- `FloorPlan.materials` switches on P.2040 walls (`wallmap/materials.py`). Without it a plan keeps the fixed reflection and the 0.4 transmission of the demo. `measure(..., snr_1m_db=...)` fixes the noise floor instead of the SNR.

## Web demo

- `demo/` is a React and Vite page from the task-web starter. It plays the walk that `experiments/demo_walk.py` exports.
- `wallmap/floorplan.py` holds the apartment (10 m by 8 m, 5 rooms, door gaps) and the walk (179 positions). Each interior wall that a path crosses multiplies the amplitude by 0.4.
- `fuse_walk` in `wallmap/fusion.py` fuses the walk. A wall needs 8 links, 1 m of spread in client distance and a support test. The spread test removes the false walls that a straight walk between parallel walls makes.
- We tuned the fusion parameters on this same walk. Do not report the demo numbers as a held-out result.
- Build on the sandbox: run `uv run python demo_walk.py` in `experiments/` (about 4 minutes), then `npm ci && npm run build` in `demo/`. The JSON goes to `demo/public/wall-mapping-assets/`, which git and the sync script ignore.
- Run `npm run audit -- <url>` before you deploy.
- Deploy: rename `dist/index.html` to `wall-mapping.html`, then rsync it and `dist/wall-mapping-assets/` to `dev-demo:dev-demo/public/`. The page is at https://dev.zyx.tw/wall-mapping.

## Verify

```sh
uv run pytest -q
```

All 12 tests must pass. For an estimator change, also run `experiments/stage2.py 20` and compare the reference row (160 MHz, 4 antennas, 30 dB) against `results/stage2/stage2.json`.
