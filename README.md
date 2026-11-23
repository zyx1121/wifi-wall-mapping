<div align="center">

<h1>wifi-wall-mapping</h1>

<p><strong>Simulation of how accurately one Wi-Fi AP finds room walls from CSI multipath.</strong></p>

</div>

![Wall detection rate against bandwidth, for 2, 4 and 8 AP antennas and 5 estimators](results/stage2/detect_vs_bw.png)

Wall detection rate for one client position at 30 dB SNR (100 random positions per point). Antennas help as much as bandwidth: with MUSIC, 8 antennas at 80 MHz find more walls than 4 antennas at 160 MHz.

Watch 1 AP map an apartment while a person walks through every room: [dev.zyx.tw/wall-mapping](https://dev.zyx.tw/wall-mapping).

A home AP sees each wall as an echo of the client. This repo measures how close to the true wall position the AP gets, as a function of bandwidth, antenna count, estimator and hardware error.

## Features

- **Synthesize CSI**: 802.11 tone grids for 80, 160 and 320 MHz at 6.425 GHz, with an exact wideband array response.
- **Compare 5 estimators**: IFFT, OMP, NOMP, joint angle and delay MUSIC, and Matrix Pencil.
- **Add hardware errors**: random packet phase, timing offset, antenna phase, AGC, 8-bit quantization and RF chain ripple.
- **Fuse client positions**: cluster candidate walls, test consistency and solve the unknown LoS length.

## Results

The reference setup is 160 MHz, 4 antennas and 30 dB.

| Stage | Question | Result |
|-------|----------|--------|
| 1 | Is the geometry exact? | Yes. Wall error with true path parameters is below 1e-14 m. |
| 1 | How exact must the path parameters be? | Median wall error stays at or below 0.3 m for length noise up to 0.5 m. |
| 2 | Which estimator, one client position? | MUSIC: median 0.04 m, p90 0.15 m, 75% of walls found. IFFT finds 24%. |
| 3 | Which hardware error hurts? | Antenna phase most. Without calibration, MUSIC finds 8% of walls instead of 71%. RF chain ripple lowers MUSIC to 52%. |
| 4 | Does fusion remove false walls? | Yes, with NOMP and known LoS length: 0.1 false walls per map at 100 positions. An unknown LoS length leaves 1.4. |

Stage 3 at the reference setup, median wall error (m) and detection rate:

| Impairment | NOMP | MUSIC |
|------------|------|-------|
| none | 0.07, 0.72 | 0.04, 0.71 |
| all, antenna phase calibrated to 5 degrees | 0.09, 0.66 | 0.08, 0.49 |
| all, antenna phase uncalibrated | 0.28, 0.18 | 0.25, 0.09 |

Random packet phase, AGC and timing offset have no effect. The estimators use delay and angle relative to the LoS path, and these errors are common to all paths.

Stage 4 in 2D with all calibrated hardware errors, NOMP, 12 random rooms. Cells: median wall error (m), detection rate, false walls per map:

| LoS length | Bandwidth | 1 position | 10 positions | 100 positions |
|------------|-----------|------------|--------------|---------------|
| known | 160 MHz | 0.12, 0.61, 10.4 | 0.07, 0.83, 3.1 | 0.03, 0.92, 0.1 |
| known | 320 MHz | 0.06, 0.83, 11.1 | 0.02, 0.97, 5.3 | 0.02, 0.97, 0.4 |
| FTM, 0.3 m noise | 320 MHz | 0.13, 0.83, 11.1 | 0.06, 0.97, 5.6 | 0.03, 0.97, 0.3 |
| unknown | 320 MHz | not solvable | 0.14, 0.86, 7.3 | 0.06, 0.92, 1.4 |

MUSIC loses walls when it fuses more positions (160 MHz: 0.42 at 1 position, 0.25 at 100). The cause is not known yet. Floor and ceiling bounces (2.5D) leave 1.7 false walls per map with NOMP at 320 MHz.

## Reproduce

```sh
uv sync
cd experiments
uv run python stage1.py
uv run python stage2.py 100
uv run python stage3.py 100
uv run python stage4.py 12
```

Each script writes JSON and PNG files to `results/<stage>/`. `demo_walk.py` writes the walk that the web demo in `demo/` plays. On 8 CPU cores, stage 2 takes 29 minutes, stage 3 takes 15 minutes and stage 4 takes 28 minutes.

## How it works

```mermaid
flowchart LR
  A[Room and client position] --> B[Image-source paths]
  B --> C[CSI with hardware errors]
  C --> D[Estimator: angle and delay per path]
  D --> E[Mirror image of the client]
  E --> F[Wall = bisector of client and image]
  F --> G[Fuse many client positions]
```

The model is 2D. The AP is on one wall with a linear array, so it measures only the cosine of the angle. Wi-Fi gives delay relative to the LoS path, so the LoS length must be known, measured by FTM or solved by fusion. Floor and ceiling bounces are optional interference (2.5D).

## For AI agents

Facts for coding agents and LLM tools. The rules for changing the code are in [AGENTS.md](AGENTS.md).

- **Summary**: Monte Carlo simulation of wall localization from Wi-Fi CSI multipath with one AP.
- **Type**: research
- **Interfaces**: none
- **Install**: `uv sync`
- **Run**: `cd experiments && uv run python stage2.py 100`
- **Verify**: `uv run pytest -q`
- **Config**: none
- **Secrets**: none
- **Limits**: 2D rectangular rooms only. No ray tracing or diffuse scattering yet. No hardware measurements yet.

## Development

```sh
git clone https://github.com/zyx1121/wifi-wall-mapping && cd wifi-wall-mapping
uv sync
uv run pytest -q
```

CI runs the alint grep rules on pull requests.
