# PGCD: Physics-Guided Conditional Diffusion for Excavation Trajectory Generation

Code, data and trained models accompanying the paper

> **Excavation trajectory generation using a physics-guided conditional diffusion model**
> Shijiang Li, Shaojie Wang, Xiangjian Bu, Tao Guo, Yongjun Pan, Liang Hou
> Xiamen University · Corresponding author: hliang@xmu.edu.cn

The repository contains the 369-condition PSO dataset, the fitted material surface of every
operating condition, the trained models and all reported results, so training and the
reported results can be checked without the raw point clouds (~23 GB).

**Confidential machine parameters.** The kinematic geometry, dynamic parameters and
constraint limits of the test excavator and the trained digging-resistance and
bucket-fill-ratio models belong to the industrial partner. They are stored only in
encrypted form (`machine_params.enc`, read by `machine_params.py`). They are needed to
evaluate trajectories (objective and constraint checks, PSO, the `test_*.py` scripts,
data generation and the field experiment), not for training, `verify_table2.py` or the
figures. The key is available from the corresponding author on reasonable request; provide
it through `PGCD_PARAMS_KEY`, `PGCD_PARAMS_KEY_FILE` or `~/.pgcd/machine_params.key`.

## Layout

| Path | Content |
|---|---|
| `pgcd_paper/` | Conditional diffusion (`cond_diffusion.py`), physics-cost surrogate and guidance (`physics_cost.py`), baselines (`fnn_baseline.py`, `pinn_baseline.py`, `cvae_baseline.py`, `pc_stomp.py`, PSO in `baselines.py`), scene / objective / dataset (`scene.py`, `objective.py`, `cond_data.py`) |
| `Trajectory_planning/Cartesian_Space/Cubic_Spline/` | Spline trajectory, composite objective and constraint checks, PSO |
| `PhysicsModel/`, `DataProcess/`, `Data_Preprocess_.../`, `Forward_model_diff/`, `Volume_prediction_model/`, `trigonometric_function.py` | Excavator kinematics and dynamics (encrypted), surface fitting, interaction features and model definitions required by the objective |
| `machine_params.py`, `machine_params.enc` | Access to the encrypted machine parameters and resistance / fill-ratio models |
| `data/scene_cache.npz` | Fitted material surface and start/end points of every condition |
| `paper_reproduction_results/` | PSO dataset (`cond_data_loo_100_40x15.npz`: 369 conditions / 222,800 samples) and the reported results (`split_results.*`, `table2_resampling.csv`, `selection_budget.npz`, `guidance_analysis.npz`) |
| `artifacts/model_runs/table2_n8/` | Trained checkpoints (seeds 0, 1), per-condition metrics and training logs. Every result in the paper uses these checkpoints |
| `scripts/` | Training, evaluation and analyses (`reproduce_models/`), data generation (`reproduce_steps/`), figures (`reproduce_plots/`) |
| `field_experiment/`, `experiment_data/` | Field experiment (Fig. 9): PGCD plan, measured log, material point cloud |

In code and result files `PGDiff` = **PGCD** and `CondDiff` = **CD**; materials
`small_stone` = fine gravel, `stone` = gravel.

## Installation

Python 3.9 or 3.10, CPU only:

```bash
pip install -r requirements.txt
```

Run all commands from the repository root.

## Reproduction

**Table 2 / Fig. 5** (N = 8, seeds 0 and 1). Evaluate the released checkpoints on the full
test sets (57 in-distribution and 88 soil conditions; `--subset-policy legacy-prefix` restricts
the evaluation to the first 48 and 60 conditions, as in an earlier evaluation):

```bash
python scripts/reproduce_models/test_pso.py
python scripts/reproduce_models/test_fnn.py
python scripts/reproduce_models/test_pinn.py
python scripts/reproduce_models/test_pc_stomp.py
python scripts/reproduce_models/test_cvae.py
python scripts/reproduce_models/test_conddiff.py
python scripts/reproduce_models/test_pgcd.py
python scripts/reproduce_models/collect_table2.py   # -> paper_reproduction_results/split_results.csv
```

To retrain first, run `train_fnn.py`, `train_pinn.py`, `train_cvae.py`,
`train_conddiff.py` and `train_pgcd.py` in the same folder (add `--force` to overwrite
the released checkpoints).

**Fig. 6** (candidate budget N): the Table 2 CD / PGCD checkpoints (seeds 0, 1) on the
Table 2 test sets, 256 candidates per condition, best-of-N subsets drawn from that pool.
At N = 1 the success equals 100 % − violation rate of Table 2 up to sampling variation:

```bash
python scripts/reproduce_models/selection_budget.py   # -> paper_reproduction_results/selection_budget.npz
```

**Figs. 7–8** (guidance strength η and guided steps K; guidance-term ablation): the Table 2
PGCD checkpoints (seeds 0, 1) on the 28 validation conditions, 96 candidates per condition.
The feasibility-only surrogate of the ablation is trained once per seed on the same training
split and stored next to the checkpoints:

```bash
python scripts/reproduce_models/guidance_analysis.py   # -> paper_reproduction_results/guidance_analysis.npz
```

**Figures 4–8**:

```bash
python scripts/reproduce_plots/run_all_plots.py      # -> paper_reproduction_results/figures/
```

**Fig. 9** (field experiment):

```bash
python field_experiment/Plot_saved_pgcd_sensor_csv_nature_axis_parallel_strong_ticks.py
```

`field_experiment/Plan_PGCD_current_material_save_sensor_csv.py` regenerates the PGCD plan
for the measured material surface.

**Dataset** (optional, slow): `python scripts/reproduce_steps/01_generate_data.py` re-runs
PSO on all conditions and records the search histories.

**Checking the reported Table 2.** The per-condition outputs of the evaluation run
reported in the paper are stored in
`artifacts/model_runs/table2_n8/*/results/seed_*/per_condition.csv`. Recompute the table
from them:

```bash
python scripts/reproduce_models/verify_table2.py
```

This reproduces every entry of `split_results.csv` exactly.

**Sampling variation.** FNN, PINN and PC-STOMP are deterministic and give identical
results on re-evaluation. CVAE, CD and PGCD sample candidates stochastically, so a fresh
evaluation draws new candidates. To quantify this, re-evaluate the released checkpoints
with fixed sampling seeds:

```bash
python scripts/reproduce_models/resample_table2.py --draws 10
```

This writes mean, standard deviation and range over the draws next to the reported values
to `paper_reproduction_results/table2_resampling.csv` (10-draw results included). A single
seeded evaluation is reproducible, e.g. `test_pgcd.py --sample-seed 0`. The `lat` rows
there reflect machine load during the resampling run, not sampling variation; the planning
times in Table 2 come from the original single-machine evaluation. Fig. 5
(`plot_fig_split_main.py`) plots this 10-draw mean ± std for CVAE / CD / PGCD; pass
`--single-draw` to plot the single evaluation in `split_results.npz` instead.

`test_pso.py` re-runs condition-specific PSO as the reference (≈ 15 min on a laptop CPU);
planning times depend on the machine.

## Data notes

* `data/scene_cache.npz` was generated from the raw platform data with
  `scripts/build_scene_cache.py`. With the raw data, set `PGCD_PLATFORM_ROOT` to it and
  `PGCD_USE_RAW_SCENES=1` to refit the surfaces from the point clouds.
* Raw point clouds and complete excavation logs are not included because of their size;
  they are available from the corresponding author on reasonable request.

## Citation

```bibtex
@article{li2026pgcd,
  title  = {Excavation trajectory generation using a physics-guided conditional diffusion model},
  author = {Li, Shijiang and Wang, Shaojie and Bu, Xiangjian and Guo, Tao and Pan, Yongjun and Hou, Liang},
  year   = {2026},
  note   = {Under review}
}
```

## License

MIT, see [LICENSE](LICENSE).
