from __future__ import annotations

"""
Plan one excavation trajectory on the current transformed material point cloud
using the PGCD conditional-diffusion generator, then draw the generated
trajectory together with the real 3D material morphology.

Default planning task:
  start Y = 2550 mm
  end   Y = 1550 mm
  end bucket posture angle = -175 deg

The script reuses the current experiment point-cloud processing code from
Compare_trajectory_pointcloud_v8.py, but does not use the old stored trajectory
planning result. The PGCD generator is trained from the cached paper
reproduction data and the final candidate is selected by a local geometric
objective computed from the current point-cloud surface.
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
PLATFORM_ROOT = PROJECT_ROOT
for _path in (PLATFORM_ROOT, PROJECT_ROOT, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    import Compare_trajectory_pointcloud_v9 as VIZ  # noqa: E402
except ModuleNotFoundError:
    try:
        import Compare_trajectory_pointcloud_v8 as VIZ  # noqa: E402
    except ModuleNotFoundError:
        import importlib.util

        _viz_candidates = [
            SCRIPT_DIR / "Compare_trajectory_pointcloud_v9.py",
            SCRIPT_DIR / "Compare_trajectory_pointcloud_v9(2).py",
            SCRIPT_DIR / "Compare_trajectory_pointcloud_v8.py",
        ]
        _viz_file = next((path for path in _viz_candidates if path.exists()), None)
        if _viz_file is None:
            raise
        _spec = importlib.util.spec_from_file_location("trajectory_viz_helpers", _viz_file)
        if _spec is None or _spec.loader is None:
            raise ImportError(f"Could not import visualization helper script: {_viz_file}")
        VIZ = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(VIZ)
from pgcd_paper.cond_diffusion import CondDiffusion  # noqa: E402
from pgcd_paper.physics_cost import make_cond_guidance, penalize, train_cond_surrogate  # noqa: E402


DATA_DIR = PROJECT_ROOT / "experiment_data"
OUTPUT_DIR = SCRIPT_DIR / "pgcd_current_material_outputs"
DEFAULT_CACHE = PROJECT_ROOT / "paper_reproduction_results" / "cond_data_loo_100_40x15.npz"

# The original sensor_data_*.csv files are comma-separated and the plotting
# code uses column 0 as time and columns 1:4 as cylinder displacements.
# The extra columns are kept as zeros so the saved planning file can be read
# by the same style of loader.
SENSOR_LIKE_NUM_COLUMNS = 14
PLANNED_SENSOR_CSV_TEMPLATE = "PGCD_planned_sensor_data_{file_index}.csv"

FILE_INDEX = 5
START_Y_MM = 2550.0
END_Y_MM = 1550.0
END_ANGLE_DEG = -175.0
MATERIAL_DIFFICULTY = 1.0

# The PGCD paper uses tau=[middle_y, middle_z, start_angle, t_end].
TAU_DIM = 4
DIFF_STEPS = 300
SURROGATE_STEPS = 300
BATCH_SIZE = 256
N_CANDIDATES = 384
ETA = 0.12
GUIDE_LAST = 40
MAX_TRAIN_SAMPLES = 60000
MAX_SURROGATE_SAMPLES = 90000
RANDOM_SEED = 7

# Current-scene bounds used to denormalize PGCD samples.
MIDDLE_Y_HALF_RANGE = 130.0
MIDDLE_Z_LOWER_OFFSET = 190.0
MIDDLE_Z_UPPER_OFFSET = 60.0
START_ANGLE_RANGE = (-90.0, -50.0)
T_END_RANGE = (4.0, 10.0)

# Local candidate-selection objective.
TARGET_MEAN_DEPTH = 95.0
TARGET_MAX_DEPTH = 180.0
MAX_ABOVE_SURFACE = 6.0
CONTROL_SURFACE_CLEARANCE = 25.0


def sensor_file_for_index(file_index: int) -> Path:
    sensor_file = DATA_DIR / "CX_test" / f"sensor_data_{file_index}.csv"
    if sensor_file.exists():
        return sensor_file

    fallback = DATA_DIR / "Data" / "CX_test" / f"sensor_data_{file_index}.csv"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Sensor file not found for index {file_index}: {sensor_file}")


def load_current_point_cloud(file_index: int) -> np.ndarray:
    point_cloud_file = DATA_DIR / "Point_cloud_material_face" / f"Point_cloud_material_face_{file_index}.ply"
    point_pose_file = DATA_DIR / "Point_cloud_material_face_pos" / f"Point_cloud_material_face_pos_{file_index}.txt"

    if not point_cloud_file.exists():
        point_cloud_file = DATA_DIR / f"Point_cloud_material_face_{file_index}.ply"
    if not point_pose_file.exists():
        point_pose_file = DATA_DIR / f"Point_cloud_material_face_pos_{file_index}.txt"

    if not point_cloud_file.exists():
        raise FileNotFoundError(f"Point-cloud file not found: {point_cloud_file}")
    if not point_pose_file.exists():
        raise FileNotFoundError(f"Point-cloud pose file not found: {point_pose_file}")

    return VIZ.get_transformed_point_cloud(point_cloud_file, point_pose_file)


def load_actual_excavation_trajectory(
    file_index: int,
    surface_profile: np.ndarray,
    duration: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sensor_file = sensor_file_for_index(file_index)
    t_all, distance_all = VIZ.data_filter_distance_only(sensor_file)
    trajectory_all = VIZ.calc_true_trajectory(distance_all)
    return VIZ.extract_excavation_window(
        t=t_all,
        distance=distance_all,
        trajectory_true=trajectory_all,
        duration=float(duration),
        surface_profile=surface_profile,
        start_time=VIZ.EXCAVATION_START_TIME,
    )


def make_surface_fit_for_requested_path(
    surface_profile: np.ndarray,
    start_y: float,
    end_y: float,
) -> dict[str, float | np.ndarray]:
    endpoint_z = VIZ.surface_z_at_y(surface_profile, np.array([start_y, end_y]))
    reference_path = np.array([[start_y, endpoint_z[0]], [end_y, endpoint_z[1]]], dtype=float)
    local_profile = VIZ.select_surface_profile_near_trajectory(surface_profile, reference_path)
    return VIZ.fit_material_surface_curve(local_profile)


def surface_z_at_y_fitted_silent(
    surface_fit: dict[str, float | np.ndarray],
    y_values: np.ndarray | float,
) -> np.ndarray:
    scalar_input = np.isscalar(y_values)
    y_query = np.asarray([y_values], dtype=float) if scalar_input else np.asarray(y_values, dtype=float)
    y_mean = float(surface_fit["y_mean"])
    y_scale = float(surface_fit["y_scale"])
    coefficients = np.asarray(surface_fit["coefficients"], dtype=float)
    z_query = np.polyval(coefficients, (y_query - y_mean) / y_scale)
    return z_query[0] if scalar_input else z_query


def make_current_scene(
    surface_fit: dict[str, float | np.ndarray],
    start_y: float,
    end_y: float,
    end_angle: float,
    material_difficulty: float,
) -> dict[str, np.ndarray | float]:
    start_z, end_z = surface_z_at_y_fitted_silent(surface_fit, np.array([start_y, end_y]))
    start_pos = np.array([start_y, start_z], dtype=float)
    end_pos = np.array([end_y, end_z], dtype=float)

    mid_y = 0.5 * (start_y + end_y)
    mid_surface_z = float(surface_z_at_y_fitted_silent(surface_fit, mid_y))
    avg_endpoint_z = 0.5 * (start_z + end_z)
    middle_z_ref = min(mid_surface_z, avg_endpoint_z)

    lb = np.array(
        [
            mid_y - MIDDLE_Y_HALF_RANGE,
            middle_z_ref - MIDDLE_Z_LOWER_OFFSET,
            START_ANGLE_RANGE[0],
            T_END_RANGE[0],
        ],
        dtype=float,
    )
    ub = np.array(
        [
            mid_y + MIDDLE_Y_HALF_RANGE,
            middle_z_ref - MIDDLE_Z_UPPER_OFFSET,
            START_ANGLE_RANGE[1],
            T_END_RANGE[1],
        ],
        dtype=float,
    )
    condition = np.array([start_pos[0], start_pos[1], end_pos[0], end_pos[1], end_angle, material_difficulty])
    return dict(start_pos=start_pos, end_pos=end_pos, lb=lb, ub=ub, condition=condition, end_angle=float(end_angle))


def denormalize_tau(tau_norm: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    return lb + np.asarray(tau_norm, dtype=float) * (ub - lb)


def tau_to_trajectory(
    tau: np.ndarray,
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    end_angle: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tau = np.asarray(tau, dtype=float).reshape(-1)
    middle = np.zeros((1, 3), dtype=float)
    middle[0, 0] = tau[0]
    middle[0, 1] = tau[1]
    middle[0, 2] = tau[3] / 2.0
    _, trajectory, velocity, acceleration, angles, _, t, control_points, dis = VIZ.trajectory_planning_cubic_spline(
        start_pos=start_pos,
        middle_pos=middle,
        end_pos=end_pos,
        start_angle=float(tau[2]),
        end_angle=float(end_angle),
        t_end=float(tau[3]),
    )
    return trajectory, velocity, acceleration, angles, control_points


def tau_to_full_plan(
    tau: np.ndarray,
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    end_angle: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return planned time, displacement, Y-Z trajectory, angles, control points and middle point.

    The returned displacement columns are the three cylinder displacements that
    can be written into sensor-like CSV columns 1, 2 and 3.
    """
    tau = np.asarray(tau, dtype=float).reshape(-1)
    middle = np.zeros((1, 3), dtype=float)
    middle[0, 0] = tau[0]
    middle[0, 1] = tau[1]
    middle[0, 2] = tau[3] / 2.0

    _, trajectory, _, _, angles, _, t, control_points, dis = VIZ.trajectory_planning_cubic_spline(
        start_pos=start_pos,
        middle_pos=middle,
        end_pos=end_pos,
        start_angle=float(tau[2]),
        end_angle=float(end_angle),
        t_end=float(tau[3]),
    )
    return t, dis, trajectory, angles, control_points, middle


def save_pgcd_plan_as_sensor_like_csv(
    tau: np.ndarray,
    scene: dict[str, np.ndarray | float],
    output_csv: Path,
    num_columns: int = SENSOR_LIKE_NUM_COLUMNS,
    time_origin: float = 0.0,
) -> dict[str, Path]:
    """Save the generated planning trajectory as a sensor_data_*.csv-like file.

    CSV layout:
      column 0   : time in seconds, starting from time_origin
      columns 1-3: planned cylinder displacements
      columns 4+ : zeros, kept only for sensor_data-like shape compatibility

    Extra helper files are also saved next to the sensor-like CSV:
      *_trajectory_yz.csv       : t, Y, Z
      *_control_points_yz.csv   : Y, Z of spline control points
      *_bucket_angle.csv        : t, bucket angle
    """
    start_pos = np.asarray(scene["start_pos"], dtype=float)
    end_pos = np.asarray(scene["end_pos"], dtype=float)
    end_angle = float(scene["end_angle"])

    t, dis, trajectory, angles, control_points, _ = tau_to_full_plan(
        tau=tau,
        start_pos=start_pos,
        end_pos=end_pos,
        end_angle=end_angle,
    )

    if num_columns < 4:
        raise ValueError(f"num_columns must be at least 4, got {num_columns}.")

    sensor_like = np.zeros((len(t), num_columns), dtype=float)
    sensor_like[:, 0] = float(time_origin) + np.asarray(t, dtype=float)
    sensor_like[:, 1:4] = np.asarray(dis, dtype=float)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_csv, sensor_like, delimiter=",", fmt="%.10f")

    trajectory_csv = output_csv.with_name(f"{output_csv.stem}_trajectory_yz.csv")
    control_points_csv = output_csv.with_name(f"{output_csv.stem}_control_points_yz.csv")
    angle_csv = output_csv.with_name(f"{output_csv.stem}_bucket_angle.csv")

    np.savetxt(
        trajectory_csv,
        np.c_[t, trajectory],
        delimiter=",",
        fmt="%.10f",
        header="t,Y,Z",
        comments="",
    )
    np.savetxt(
        control_points_csv,
        control_points,
        delimiter=",",
        fmt="%.10f",
        header="Y,Z",
        comments="",
    )
    np.savetxt(
        angle_csv,
        np.c_[t, angles],
        delimiter=",",
        fmt="%.10f",
        header="t,bucket_angle_deg",
        comments="",
    )

    return {
        "sensor_like_csv": output_csv,
        "trajectory_yz_csv": trajectory_csv,
        "control_points_yz_csv": control_points_csv,
        "bucket_angle_csv": angle_csv,
    }


def local_geometry_objective(
    tau: np.ndarray,
    scene: dict[str, np.ndarray | float],
    surface_fit: dict[str, float | np.ndarray],
) -> tuple[float, dict[str, float]]:
    start_pos = np.asarray(scene["start_pos"], dtype=float)
    end_pos = np.asarray(scene["end_pos"], dtype=float)
    end_angle = float(scene["end_angle"])

    try:
        trajectory, velocity, acceleration, angles, control_points = tau_to_trajectory(tau, start_pos, end_pos, end_angle)
    except Exception:
        return np.inf, {}

    if not np.all(np.isfinite(trajectory)) or not np.all(np.isfinite(velocity)):
        return np.inf, {}

    surface_z = surface_z_at_y_fitted_silent(surface_fit, trajectory[:, 0])
    depth = surface_z - trajectory[:, 1]
    above = np.maximum(-depth, 0.0)
    positive_depth = np.maximum(depth, 0.0)
    cp_surface_z = surface_z_at_y_fitted_silent(surface_fit, control_points[:, 0])
    cp_clearance = cp_surface_z - control_points[:, 1]

    if np.nanmax(above) > MAX_ABOVE_SURFACE:
        above_penalty = 60.0 * float(np.mean(above**2))
    else:
        above_penalty = 8.0 * float(np.mean(above**2))

    mean_depth = float(np.nanmean(positive_depth))
    max_depth = float(np.nanmax(positive_depth))
    active_ratio = float(np.mean(depth > 10.0))
    length = float(np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)))
    roughness = float(np.nanmean(np.linalg.norm(acceleration, axis=1) ** 2))

    depth_cost = ((mean_depth - TARGET_MEAN_DEPTH) / 70.0) ** 2
    max_depth_cost = ((max_depth - TARGET_MAX_DEPTH) / 110.0) ** 2
    active_cost = (active_ratio - 0.72) ** 2
    cp_cost = float(np.mean(np.maximum(CONTROL_SURFACE_CLEARANCE - cp_clearance, 0.0) ** 2)) / 1200.0
    length_cost = ((length - 1180.0) / 600.0) ** 2
    roughness_cost = roughness * 1e-5
    angle_cost = ((float(tau[2]) + 70.0) / 18.0) ** 2
    time_cost = (float(tau[3]) - 7.0) ** 2 / 6.0

    total = (
        above_penalty
        + 1.6 * depth_cost
        + 0.8 * max_depth_cost
        + 0.7 * active_cost
        + 0.8 * cp_cost
        + 0.3 * length_cost
        + 0.2 * roughness_cost
        + 0.45 * angle_cost
        + 0.65 * time_cost
    )
    metrics = dict(
        mean_depth=mean_depth,
        max_depth=max_depth,
        active_ratio=active_ratio,
        path_length=length,
        roughness=roughness,
        start_angle=float(tau[2]),
        t_end=float(tau[3]),
    )
    return float(total), metrics


def load_pgcd_training_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"PGCD cache not found: {path}\n"
            "Run the paper reproduction data step first, or pass --cache to an existing cond_data_*.npz file."
        )
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def sample_rows(mask: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    idx = np.where(mask)[0]
    if len(idx) <= max_rows:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=max_rows, replace=False))


def nearest_condition_candidates(
    data: dict[str, np.ndarray],
    condition: np.ndarray,
    n: int,
) -> np.ndarray:
    conds = np.asarray(data["conds"], dtype=float)
    cm = conds.mean(axis=0)
    cs = conds.std(axis=0) + 1e-9
    dist = np.linalg.norm((conds - condition) / cs, axis=1)
    nearest_cids = np.argsort(dist)[: min(8, len(dist))]

    samp_cond = data["samp_cond"].astype(int)
    J = np.asarray(data["samp_J"], dtype=float)
    mask = np.isin(samp_cond, nearest_cids) & np.isfinite(J)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return np.empty((0, TAU_DIM), dtype=float)

    idx = idx[np.argsort(J[idx])[:n]]
    return np.asarray(data["samp_tau"][idx], dtype=float)


def generate_pgcd_candidates(
    data: dict[str, np.ndarray],
    condition: np.ndarray,
    n_candidates: int,
    diff_steps: int,
    surrogate_steps: int,
    batch_size: int,
    eta: float,
    guide_last: int,
    max_train_samples: int,
    max_surrogate_samples: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    import torch

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    J = np.asarray(data["samp_J"], dtype=float)
    tau = np.asarray(data["samp_tau"], dtype=float)
    cond = np.asarray(data["conds"], dtype=float)[data["samp_cond"].astype(int)]
    feasible = np.isfinite(J)

    train_idx = sample_rows(feasible, max_train_samples, seed)
    surr_idx = sample_rows(np.ones_like(feasible, dtype=bool), max_surrogate_samples, seed + 1)
    rng.shuffle(train_idx)
    rng.shuffle(surr_idx)

    t0 = time.time()
    diff = CondDiffusion().fit(
        tau[train_idx],
        cond[train_idx],
        J[train_idx],
        steps=diff_steps,
        bs=batch_size,
        lr=8e-4,
        bc_temp=0.3,
        verbose=False,
    )
    surrogate, _ = train_cond_surrogate(
        diff.std_tau(tau[surr_idx]),
        diff.std_cond(cond[surr_idx]),
        penalize(J[surr_idx]),
        steps=surrogate_steps,
        bs=batch_size,
        lr=1e-3,
        verbose=False,
    )
    guide = make_cond_guidance(surrogate)
    generated = diff.sample(condition, n=n_candidates, guidance=guide, eta=eta, guide_last=guide_last)

    nearby = nearest_condition_candidates(data, condition, max(32, n_candidates // 4))
    if nearby.size:
        generated = np.vstack([generated, nearby])
    return np.clip(generated, 0.0, 1.0), time.time() - t0


def select_best_candidate(
    tau_norm_candidates: np.ndarray,
    scene: dict[str, np.ndarray | float],
    surface_fit: dict[str, float | np.ndarray],
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    lb = np.asarray(scene["lb"], dtype=float)
    ub = np.asarray(scene["ub"], dtype=float)
    tau_candidates = denormalize_tau(tau_norm_candidates, lb, ub)

    best_i = -1
    best_score = np.inf
    best_metrics: dict[str, float] = {}
    for i, tau in enumerate(tau_candidates):
        score, metrics = local_geometry_objective(tau, scene, surface_fit)
        tau_norm = np.asarray(tau_norm_candidates[i], dtype=float)
        edge_low = np.maximum(0.08 - tau_norm, 0.0) / 0.08
        edge_high = np.maximum(tau_norm - 0.92, 0.0) / 0.08
        score += 1.5 * float(np.sum(edge_low**2 + edge_high**2))
        if score < best_score:
            best_i = i
            best_score = score
            best_metrics = metrics

    if best_i < 0 or not np.isfinite(best_score):
        raise RuntimeError("No valid PGCD candidate could be selected for the current material.")
    return tau_norm_candidates[best_i], tau_candidates[best_i], float(best_score), best_metrics


def plot_pgcd_plan(
    point_cloud: np.ndarray,
    surface_fit: dict[str, float | np.ndarray],
    scene: dict[str, np.ndarray | float],
    tau: np.ndarray,
    actual_trajectory: np.ndarray,
    output_dir: Path,
) -> None:
    VIZ.set_plot_style()
    start_pos = np.asarray(scene["start_pos"], dtype=float)
    end_pos = np.asarray(scene["end_pos"], dtype=float)
    end_angle = float(scene["end_angle"])
    trajectory, _, _, _, control_points = tau_to_trajectory(tau, start_pos, end_pos, end_angle)
    cropped = VIZ.crop_point_cloud_for_trajectory(point_cloud, actual_trajectory, trajectory)

    fig = plt.figure(figsize=(17.0 / 2.54, 6.0 / 2.54))
    ax2d = fig.add_axes([0.07, 0.18, 0.39, 0.52])
    ax3d = fig.add_axes([0.56, 0.18, 0.39, 0.52], projection="3d")

    y_curve = np.linspace(min(start_pos[0], end_pos[0]) - 120.0, max(start_pos[0], end_pos[0]) + 120.0, 400)
    z_curve = surface_z_at_y_fitted_silent(surface_fit, y_curve)

    ax2d.plot(y_curve, z_curve, "-.", color="0.35", linewidth=1.1, label="Material surface")
    ax2d.plot(trajectory[:, 0], trajectory[:, 1], "-", color="#1f77b4", linewidth=1.55, label="PGCD trajectory")
    ax2d.plot(
        actual_trajectory[:, 0],
        actual_trajectory[:, 1],
        "-",
        color="#ff7f0e",
        linewidth=1.25,
        label="Actual trajectory",
        zorder=4,
    )
    ax2d.scatter(control_points[:, 0], control_points[:, 1], color="red", s=10, label="Control points", zorder=5)
    ax2d.scatter(start_pos[0], start_pos[1], color="green", s=16, label="Start point", zorder=6)
    ax2d.scatter(end_pos[0], end_pos[1], color="blue", s=16, label="End point", zorder=6)
    ax2d.set_xlabel("Y (mm)")
    ax2d.set_ylabel("Z (mm)")
    ax2d.text(-0.12, 1.00, "(a)", transform=ax2d.transAxes, ha="left", va="top", fontweight="bold")
    ax2d.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.48, 1.03), ncol=2, borderaxespad=0.0)

    if cropped.size:
        xyz, rgb = VIZ._split_xyz_rgb(cropped)
        scatter_kwargs = dict(s=0.7, alpha=0.20, linewidths=0, label="Material point cloud")
        scatter_kwargs["c"] = rgb if rgb is not None else "0.72"
        ax3d.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], **scatter_kwargs)
    else:
        xyz = np.c_[np.zeros(len(trajectory)), trajectory]

    x0 = np.zeros(trajectory.shape[0])
    xa = np.zeros(actual_trajectory.shape[0])
    xcp = np.zeros(control_points.shape[0])
    ax3d.plot(x0, trajectory[:, 0], trajectory[:, 1], "-", color="#1f77b4", linewidth=1.2, label="PGCD trajectory")
    ax3d.plot(
        xa,
        actual_trajectory[:, 0],
        actual_trajectory[:, 1],
        "-",
        color="#ff7f0e",
        linewidth=1.05,
        label="Actual trajectory",
    )
    ax3d.scatter(xcp, control_points[:, 0], control_points[:, 1], color="red", s=8, label="Control points")
    try:
        VIZ._set_3d_limits_and_aspect(ax3d, xyz[:, :3], actual_trajectory, trajectory, control_points, box_zoom=2.05)
    except TypeError:
        VIZ._set_3d_limits_and_aspect(ax3d, xyz[:, :3], actual_trajectory, trajectory, control_points)
    ax3d.view_init(elev=VIZ.VIEW_3D_ELEV, azim=VIZ.VIEW_3D_AZIM)
    ax3d.set_xlabel("")
    ax3d.set_ylabel("")
    ax3d.set_zlabel("")
    VIZ._format_3d_axes(ax3d)
    ax3d.text2D(-0.12, 1.00, "(b)", transform=ax3d.transAxes, ha="left", va="top", fontweight="bold")
    ax3d.text2D(0.18, 0.08, "X (mm)", transform=ax3d.transAxes, rotation=-15, ha="center", va="center")
    ax3d.text2D(0.84, 0.04, "Y (mm)", transform=ax3d.transAxes, rotation=13, ha="center", va="center")
    ax3d.text2D(1.00, 0.50, "Z (mm)", transform=ax3d.transAxes, rotation=90, ha="center", va="center")
    ax3d.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.52, 1.03), ncol=2, borderaxespad=0.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_svg = output_dir / "PGCD_current_material_plan.svg"
    if hasattr(VIZ, "save_figure"):
        VIZ.save_figure(fig, out_svg, pad_inches=0.04)
    else:
        fig.savefig(out_svg, format="svg", bbox_inches="tight", pad_inches=0.04)
    np.save(output_dir / "PGCD_current_material_tau.npy", np.asarray(tau, dtype=float))
    np.save(output_dir / "PGCD_current_material_trajectory.npy", np.asarray(trajectory, dtype=float))
    np.save(output_dir / "PGCD_current_material_control_points.npy", np.asarray(control_points, dtype=float))
    np.save(output_dir / "actual_excavation_trajectory.npy", np.asarray(actual_trajectory, dtype=float))
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan a PGCD trajectory on the current material point cloud.")
    parser.add_argument("--file-index", type=int, default=FILE_INDEX, help="Current material point-cloud index.")
    parser.add_argument("--start-y", type=float, default=START_Y_MM, help="Excavation start Y coordinate in mm.")
    parser.add_argument("--end-y", type=float, default=END_Y_MM, help="Excavation end Y coordinate in mm.")
    parser.add_argument("--end-angle", type=float, default=END_ANGLE_DEG, help="End bucket posture angle in degrees.")
    parser.add_argument("--material-difficulty", type=float, default=MATERIAL_DIFFICULTY,
                        help="PGCD condition material difficulty: sand/soil=1, small stone=2, stone=3.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="Cached PGCD training data .npz.")
    parser.add_argument("--n-candidates", type=int, default=N_CANDIDATES, help="Number of PGCD candidates to sample.")
    parser.add_argument("--diff-steps", type=int, default=DIFF_STEPS, help="Conditional diffusion training steps.")
    parser.add_argument("--surrogate-steps", type=int, default=SURROGATE_STEPS, help="Guidance surrogate training steps.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--eta", type=float, default=ETA, help="Physics-guidance strength.")
    parser.add_argument("--guide-last", type=int, default=GUIDE_LAST, help="Guidance is applied in last DDPM steps.")
    parser.add_argument("--max-train-samples", type=int, default=MAX_TRAIN_SAMPLES)
    parser.add_argument("--max-surrogate-samples", type=int, default=MAX_SURROGATE_SAMPLES)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--planned-sensor-csv", type=Path, default=None,
                        help="Optional output path for the planned sensor-like CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    point_cloud = load_current_point_cloud(args.file_index)
    surface_profile = VIZ.build_surface_profile_from_point_cloud(point_cloud)
    surface_fit = make_surface_fit_for_requested_path(surface_profile, args.start_y, args.end_y)
    scene = make_current_scene(surface_fit, args.start_y, args.end_y, args.end_angle, args.material_difficulty)

    data = load_pgcd_training_cache(args.cache)
    tau_norm_candidates, train_time = generate_pgcd_candidates(
        data=data,
        condition=np.asarray(scene["condition"], dtype=float),
        n_candidates=args.n_candidates,
        diff_steps=args.diff_steps,
        surrogate_steps=args.surrogate_steps,
        batch_size=args.batch_size,
        eta=args.eta,
        guide_last=args.guide_last,
        max_train_samples=args.max_train_samples,
        max_surrogate_samples=args.max_surrogate_samples,
        seed=args.seed,
    )
    best_tau_norm, best_tau, best_score, metrics = select_best_candidate(tau_norm_candidates, scene, surface_fit)
    t_actual, dis_actual, actual_trajectory = load_actual_excavation_trajectory(
        file_index=args.file_index,
        surface_profile=surface_profile,
        duration=float(best_tau[3]),
    )
    plot_pgcd_plan(point_cloud, surface_fit, scene, best_tau, actual_trajectory, args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "actual_excavation_time.npy", np.asarray(t_actual, dtype=float))
    np.save(args.output_dir / "actual_excavation_displacement.npy", np.asarray(dis_actual, dtype=float))

    planned_sensor_csv = (
        args.planned_sensor_csv
        if args.planned_sensor_csv is not None
        else args.output_dir / PLANNED_SENSOR_CSV_TEMPLATE.format(file_index=args.file_index)
    )
    saved_plan_files = save_pgcd_plan_as_sensor_like_csv(
        tau=best_tau,
        scene=scene,
        output_csv=planned_sensor_csv,
    )

    print("PGCD current-material planning complete")
    print(f"file_index: {args.file_index}")
    print(f"start_pos [Y,Z]: {np.asarray(scene['start_pos'])}")
    print(f"end_pos   [Y,Z]: {np.asarray(scene['end_pos'])}")
    print(f"end_angle: {args.end_angle}")
    print(f"condition: {np.asarray(scene['condition'])}")
    print(f"bounds lb: {np.asarray(scene['lb'])}")
    print(f"bounds ub: {np.asarray(scene['ub'])}")
    print(f"candidate count: {len(tau_norm_candidates)}")
    print(f"PGCD train/sample time: {train_time:.2f} s")
    print(f"best tau_norm [middle_y,middle_z,start_angle,t_end]: {best_tau_norm}")
    print(f"best tau      [middle_y,middle_z,start_angle,t_end]: {best_tau}")
    print(f"actual trajectory samples: {len(actual_trajectory)}")
    print(f"local geometry score: {best_score:.6f}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6g}")
    print(f"Saved: {args.output_dir / 'PGCD_current_material_plan.svg'}")
    for label, path in saved_plan_files.items():
        print(f"Saved {label}: {path}")


if __name__ == "__main__":
    main()
