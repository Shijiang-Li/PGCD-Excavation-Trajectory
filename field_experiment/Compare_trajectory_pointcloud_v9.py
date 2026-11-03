from __future__ import annotations

"""
Compare real bucket-tip trajectory and mined/planned trajectory directly on the
real transformed 3D point cloud.

Key points:
1. plot_trajectory still uses the real 3D point cloud as the background, and
   can optionally draw a fitted/extrapolated Y-Z material-surface curve in the 2D plot.
2. The material background is the real 3D point cloud cropped around the
   excavation trajectory instead of the whole point cloud.
3. Original point-cloud RGB colors are kept when available.
4. Start/end points of the planned excavation trajectory are snapped to the
   material surface by Y coordinate. By default, a fitted surface curve is used
   for endpoint Z so that missing end-side point cloud can be extrapolated.
5. No resistance, volume, energy, or trajectory-error metrics are computed.
6. The optional 3D view supports controllable figure size and box aspect ratio.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from scipy.interpolate import CubicSpline

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
PLATFORM_ROOT = PROJECT_ROOT
for _path in (PLATFORM_ROOT, PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

DATA_DIR = PROJECT_ROOT / "experiment_data"
OUTPUT_DIR = SCRIPT_DIR / "calc_excavation_outputs"

from PhysicsModel.Kinematics_0314 import Kinematic  # noqa: E402


# =========================
# User-adjustable settings
# =========================
FILE_INDEX = 5
SKIP_ROWS = 200
END_ANGLE = -175

# If you know the excavation start time in the filtered sensor record, set it here.
# If it is None, the script finds the first bucket-tip penetration into the
# point-cloud surface. If no reliable contact is found, it falls back to the
# largest-motion window.
EXCAVATION_START_TIME: float | None = None

# Parameters used to obtain a Y-Z material-surface profile from the real 3D cloud.
# This is NOT polynomial fitting: it only extracts a narrow X-direction section,
# bins real points by Y, and interpolates between the binned cloud samples.
SURFACE_X_CENTER = 0.0
SURFACE_X_HALF_WIDTH = 12.0
SURFACE_Y_BIN_SIZE = 5.0
SURFACE_MIN_POINTS_PER_BIN = 3
SURFACE_Z_PERCENTILE = 50.0  # 50=median; use 90/95 if lower outliers exist.
EXCAVATION_DEPTH_THRESHOLD_MM = 0.0

# Point-cloud crop range around the trajectory, in millimetres.
POINT_CLOUD_X_CENTER = 0.0
POINT_CLOUD_X_MARGIN = 180.0
POINT_CLOUD_Y_MARGIN = 320.0
POINT_CLOUD_Z_MARGIN = 260.0
MAX_POINT_CLOUD_POINTS = 8000
RANDOM_SEED = 0

# Whether to also save a 3D view. The 2D Y-Z projection is the closest match to
# the original plot_trajectory figure.
SAVE_3D_VIEW = True

# 3D display controls.
# FIGURE_3D_SIZE_CM controls the saved/onscreen figure width and height.
# When USE_DATA_ASPECT_3D=True, the 3D axes box aspect follows the real data
# ranges in X/Y/Z, which avoids visual stretching or compression.
# When USE_DATA_ASPECT_3D=False, MANUAL_BOX_ASPECT_3D is used directly as the
# display ratio of X:Y:Z. Example: (1.0, 1.6, 0.8) makes Y visually longer.
FIGURE_3D_SIZE_CM = (10.0, 6.0)
USE_DATA_ASPECT_3D = False
MANUAL_BOX_ASPECT_3D = (1.4, 1.6, 0.8)
LIMITS_3D_MARGIN_RATIO = 0.01

# 3D view angle controls.
# elev: vertical viewing angle, larger means looking more from above.
# azim: horizontal rotation angle around the Z axis.
VIEW_3D_ELEV = 7
VIEW_3D_AZIM = -40

# 3D axis/tick layout controls. Use these when tick labels overlap.
MAX_TICKS_3D = 5
TICK_LABEL_SIZE_3D = 8
AXIS_LABEL_SIZE_3D = 8
AXIS_LABEL_PAD_3D = 1
TICK_PAD_3D = -1
TICK_FORMAT_3D = "%.0f"

# Make the 3D axes fill more of the saved figure.
# Values are in figure-relative coordinates: left, bottom, width, height.
# If labels get clipped, reduce width/height slightly or increase SAVE_PAD_INCHES_3D.
AXES_POSITION_3D = (-0.12, -0.05, 1, 1)
SAVE_PAD_INCHES_3D = 0.01

# Optional: hide one axis tick labels if a specific view still overlaps.
HIDE_X_TICK_LABELS_3D = False
HIDE_Y_TICK_LABELS_3D = False
HIDE_Z_TICK_LABELS_3D = False

# Turn this on when you are choosing a good viewing angle. It will save a small
# set of candidate 3D views, then you can copy the best elev/azim values above.
SAVE_3D_VIEW_CANDIDATES = False
CANDIDATE_3D_VIEWS = [
    (15, -75),
    (20, -60),
    (25, -45),
    (30, -60),
    (20, -120),
    (35, -35),
]

# 2D figure layout controls.
FIGURE_2D_SIZE_CM = (10.0, 6.0)
LEGEND_OUTSIDE_2D = True
LEGEND_NCOL_2D = 3

# Point-cloud transparency. The point cloud is now hidden in the 2D figure and
# still shown in the optional 3D figure.
DRAW_POINT_CLOUD_2D = False
POINT_CLOUD_ALPHA_2D = 0.28
POINT_CLOUD_ALPHA_3D = 0.20

# 2D fitted material-surface curve.
# The curve is fitted from the real point-cloud Y-Z surface profile and is only
# drawn in the 2D Y-Z trajectory figure. It is not used as a 3D point-cloud fill.
DRAW_FITTED_SURFACE_2D = True
# Use a 4th-order polynomial, matching the original material-surface fitting style.
# The fit is NOT built from the whole point-cloud profile; it is built only from
# the point-cloud surface samples near the extracted excavation trajectory.
FITTED_SURFACE_DEGREE = 4
FITTED_SURFACE_NUM_POINTS = 400
FITTED_SURFACE_Y_MARGIN = 80.0
FITTED_SURFACE_EXTRAPOLATION_MM = 350.0
FITTED_SURFACE_LINEWIDTH = 1.1
FITTED_SURFACE_FIT_Y_MARGIN = 120.0
FITTED_SURFACE_MIN_FIT_POINTS = 8

# If True, planned trajectory start/end Z values are also taken from the fitted
# curve. This is useful when the endpoint is outside the observed point-cloud
# surface range and direct interpolation would clamp to the nearest observed bin.
USE_FITTED_SURFACE_FOR_ENDPOINTS = True


def _read_ply_points_and_colors(point_cloud_file: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read xyz points and optional RGB colors from a PLY file.

    Colors are returned as float RGB in [0, 1] when they exist. If the PLY has no
    color attributes, the second return value is None.
    """
    try:
        import open3d as o3d

        point_cloud = o3d.io.read_point_cloud(str(point_cloud_file))
        points = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors) if point_cloud.has_colors() else None
        if colors is not None and colors.size == 0:
            colors = None
        return points, colors
    except ModuleNotFoundError:
        pass

    header_lines = 0
    vertex_count = None
    properties: list[str] = []
    with open(point_cloud_file, "r", encoding="utf-8") as fh:
        for line in fh:
            header_lines += 1
            stripped = line.strip()
            if stripped.startswith("element vertex"):
                vertex_count = int(stripped.split()[-1])
            elif stripped.startswith("property"):
                properties.append(stripped.split()[-1])
            elif stripped == "end_header":
                break

    if vertex_count is None:
        raise ValueError(f"PLY file has no vertex count: {point_cloud_file}")

    xyz_cols = [properties.index(name) for name in ("x", "y", "z")]

    rgb_cols = None
    for names in (("red", "green", "blue"), ("r", "g", "b")):
        if all(name in properties for name in names):
            rgb_cols = [properties.index(name) for name in names]
            break

    usecols = xyz_cols if rgb_cols is None else xyz_cols + rgb_cols
    data = np.loadtxt(point_cloud_file, skiprows=header_lines, max_rows=vertex_count, usecols=usecols)
    data = np.atleast_2d(data)

    points = data[:, :3]
    colors = None
    if rgb_cols is not None:
        colors = data[:, 3:6].astype(float)
        if np.nanmax(colors) > 1.0:
            colors = colors / 255.0
        colors = np.clip(colors, 0.0, 1.0)

    return points, colors


def _combine_xyz_rgb(points: np.ndarray, colors: np.ndarray | None) -> np.ndarray:
    """Return Nx3 if colors are absent, otherwise Nx6 [X, Y, Z, R, G, B]."""
    if colors is None:
        return points
    if len(points) != len(colors):
        raise ValueError("points and colors must have the same length.")
    return np.c_[points, colors]


def _as_xyz_points(point_cloud: np.ndarray) -> np.ndarray:
    """Convert either 4xN/Nx4 homogeneous point cloud or Nx3/Nx6 point cloud to Nx3 xyz."""
    if point_cloud.ndim != 2:
        raise ValueError(f"point_cloud must be a 2D array, got shape {point_cloud.shape}.")

    if point_cloud.shape[0] in (3, 4) and point_cloud.shape[1] > point_cloud.shape[0]:
        points = point_cloud[:3, :].T
    elif point_cloud.shape[1] >= 3:
        points = point_cloud[:, :3]
    else:
        raise ValueError(f"Unsupported point_cloud shape: {point_cloud.shape}.")
    return points


def _split_xyz_rgb(point_cloud: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Split Nx3/Nx6 or homogeneous 4xN cloud into xyz and optional RGB colors."""
    points = _as_xyz_points(point_cloud)
    colors = None

    if point_cloud.ndim == 2 and point_cloud.shape[1] >= 6:
        colors = point_cloud[:, 3:6].astype(float)
        if colors.size > 0 and np.nanmax(colors) > 1.0:
            colors = colors / 255.0
        colors = np.clip(colors, 0.0, 1.0)

    return points, colors


def trans_points_cloud_with_color(point_cloud_file: Path, db_x: float, dg_x: float, cd_x: float) -> np.ndarray:
    """Transform point cloud into the excavator/world coordinate system and keep RGB if present."""
    points, colors = _read_ply_points_and_colors(point_cloud_file)
    points[:, 0] = points[:, 0] - 60

    # Keep the same x-direction pre-filter as the original script. Apply the
    # identical mask to color rows so RGB stays aligned with xyz.
    condition_x = np.logical_and(points[:, 0] >= -200, points[:, 0] <= 200)
    filtered_points = points[condition_x]
    filtered_colors = colors[condition_x] if colors is not None else None

    p_cloud = np.c_[filtered_points, np.ones(filtered_points.shape[0])].T
    transformed = Kinematic().plyTrans(p_cloud, 90, db_x, dg_x, cd_x)
    transformed_points = _as_xyz_points(transformed)
    return _combine_xyz_rgb(transformed_points, filtered_colors)


def get_transformed_point_cloud(point_cloud_file: Path, point_cloud_position_file: Path) -> np.ndarray:
    """Load point cloud and transform it using the averaged point-cloud pose record."""
    point_cloud_position_distance = np.loadtxt(point_cloud_position_file)
    db_x = np.average(point_cloud_position_distance[:, 13])
    dg_x = np.average(point_cloud_position_distance[:, 14])
    cd_x = np.average(point_cloud_position_distance[:, 15])
    return trans_points_cloud_with_color(point_cloud_file, db_x=db_x, dg_x=dg_x, cd_x=cd_x)


def data_filter_distance_only(sensor_file: Path, skip_rows: int = SKIP_ROWS) -> tuple[np.ndarray, np.ndarray]:
    """Read only time and cylinder displacement; no pressure/resistance processing."""
    t = np.loadtxt(open(sensor_file, "rb"), delimiter=",", skiprows=skip_rows, usecols=[0])
    distance = np.loadtxt(open(sensor_file, "rb"), delimiter=",", skiprows=skip_rows, usecols=[1, 2, 3])
    return t - t[0], distance


def calc_true_trajectory(distance: np.ndarray) -> np.ndarray:
    """Calculate the real bucket-tip Y-Z trajectory from the measured cylinder displacements."""
    kne = Kinematic()
    pos = kne.kine_qdTowz_batch(90, distance[:, 0], distance[:, 1], distance[:, 2])
    return pos[:, [1, 2]]


def build_surface_profile_from_point_cloud(
    point_cloud: np.ndarray,
    x_center: float = SURFACE_X_CENTER,
    x_half_width: float = SURFACE_X_HALF_WIDTH,
    y_bin_size: float = SURFACE_Y_BIN_SIZE,
    min_points_per_bin: int = SURFACE_MIN_POINTS_PER_BIN,
    z_percentile: float = SURFACE_Z_PERCENTILE,
) -> np.ndarray:
    """
    Extract a Y-Z surface profile from the real 3D point cloud without fitting.

    The output has shape (n, 2), columns [Y, Z_surface]. It uses a narrow slice
    around X=x_center, groups points by Y bins, and stores a percentile Z from
    the actual cloud points in each bin.
    """
    points = _as_xyz_points(point_cloud)
    section = points[(points[:, 0] >= x_center - x_half_width) & (points[:, 0] <= x_center + x_half_width)]

    # If the X slice is too narrow for a sparse point cloud, expand it once.
    if section.shape[0] < 100:
        expanded_half_width = max(2.0 * x_half_width, 30.0)
        section = points[
            (points[:, 0] >= x_center - expanded_half_width) & (points[:, 0] <= x_center + expanded_half_width)
        ]
        print(
            f"Warning: surface X slice was sparse; expanded half width to {expanded_half_width:.1f} mm "
            f"and kept {section.shape[0]} points."
        )

    if section.shape[0] == 0:
        raise ValueError("No point-cloud points available for building the material surface profile.")

    y = section[:, 1]
    z = section[:, 2]
    y_min = np.floor(np.min(y) / y_bin_size) * y_bin_size
    y_max = np.ceil(np.max(y) / y_bin_size) * y_bin_size
    bins = np.arange(y_min, y_max + y_bin_size, y_bin_size)
    if bins.size < 2:
        raise ValueError("The point-cloud Y range is too small to build a surface profile.")

    bin_ids = np.digitize(y, bins) - 1
    profile_rows: list[tuple[float, float]] = []
    for bin_id in range(len(bins) - 1):
        mask = bin_ids == bin_id
        if np.count_nonzero(mask) < min_points_per_bin:
            continue
        y_mid = 0.5 * (bins[bin_id] + bins[bin_id + 1])
        z_surface = float(np.percentile(z[mask], z_percentile))
        profile_rows.append((float(y_mid), z_surface))

    if len(profile_rows) < 2:
        raise ValueError(
            "Too few point-cloud Y bins to build a material surface profile. "
            "Increase SURFACE_X_HALF_WIDTH or SURFACE_Y_BIN_SIZE."
        )

    return np.asarray(profile_rows)


def surface_z_at_y(surface_profile: np.ndarray, y_values: np.ndarray | float) -> np.ndarray:
    """Return surface Z at Y by linear interpolation between real binned cloud samples."""
    scalar_input = np.isscalar(y_values)
    y_query = np.asarray([y_values], dtype=float) if scalar_input else np.asarray(y_values, dtype=float)

    order = np.argsort(surface_profile[:, 0])
    y_profile = surface_profile[order, 0]
    z_profile = surface_profile[order, 1]

    z_query = np.interp(y_query, y_profile, z_profile)

    # Warn when the requested point is outside the observed cloud profile range.
    outside = (y_query < y_profile[0]) | (y_query > y_profile[-1])
    if np.any(outside):
        print(
            "Warning: some trajectory Y values are outside the point-cloud surface range; "
            "nearest endpoint interpolation was used."
        )

    return z_query[0] if scalar_input else z_query




def select_surface_profile_near_trajectory(
    surface_profile: np.ndarray,
    trajectory_yz: np.ndarray,
    y_margin: float = FITTED_SURFACE_FIT_Y_MARGIN,
    min_points: int = FITTED_SURFACE_MIN_FIT_POINTS,
) -> np.ndarray:
    """Select real point-cloud surface-profile samples near the excavation trajectory.

    This keeps the fitted material-surface curve local to the current excavation
    region instead of fitting the whole pile/profile. If the first local window is
    too sparse, the window is expanded gradually, but it still remains centered on
    the trajectory Y range.
    """
    if surface_profile.ndim != 2 or surface_profile.shape[1] < 2:
        raise ValueError(f"surface_profile must have shape (n, 2), got {surface_profile.shape}.")
    if trajectory_yz.ndim != 2 or trajectory_yz.shape[1] < 2:
        raise ValueError(f"trajectory_yz must have shape (n, 2), got {trajectory_yz.shape}.")

    finite_mask = np.isfinite(surface_profile[:, 0]) & np.isfinite(surface_profile[:, 1])
    profile = surface_profile[finite_mask]
    if profile.shape[0] < min_points:
        print(
            "Warning: whole surface profile is sparse; using all available profile samples "
            "for the fitted curve."
        )
        return profile

    y_min = float(np.nanmin(trajectory_yz[:, 0]))
    y_max = float(np.nanmax(trajectory_yz[:, 0]))

    # Expand only if necessary. This avoids accidentally fitting distant, unrelated
    # pile regions while still preventing a failed 4th-order polynomial fit.
    margins_to_try = [y_margin, 1.5 * y_margin, 2.0 * y_margin, 3.0 * y_margin]
    selected = profile
    used_margin = margins_to_try[-1]
    for margin in margins_to_try:
        mask = (profile[:, 0] >= y_min - margin) & (profile[:, 0] <= y_max + margin)
        candidate = profile[mask]
        if candidate.shape[0] >= min_points:
            selected = candidate
            used_margin = margin
            break
    else:
        print(
            "Warning: not enough local point-cloud surface samples near the excavation trajectory; "
            "using the closest profile samples instead."
        )
        y_center = 0.5 * (y_min + y_max)
        order = np.argsort(np.abs(profile[:, 0] - y_center))
        selected = profile[order[: min(max(min_points, FITTED_SURFACE_DEGREE + 1), profile.shape[0])]]

    print(
        f"Fitted surface uses {selected.shape[0]} local point-cloud profile samples "
        f"near trajectory Y range [{y_min:.1f}, {y_max:.1f}] mm "
        f"with margin {used_margin:.1f} mm."
    )
    return selected[np.argsort(selected[:, 0])]


def fit_material_surface_curve(
    surface_profile: np.ndarray,
    degree: int = FITTED_SURFACE_DEGREE,
) -> dict[str, float | np.ndarray]:
    """Fit a polynomial Z=f(Y) curve from the real point-cloud surface profile.

    The fit uses normalized Y values for numerical stability. The returned curve
    can be evaluated outside the observed profile range for limited extrapolation.
    """
    finite_mask = np.isfinite(surface_profile[:, 0]) & np.isfinite(surface_profile[:, 1])
    profile = surface_profile[finite_mask]
    if profile.shape[0] < 2:
        raise ValueError("Too few finite surface-profile points for fitting.")

    order = np.argsort(profile[:, 0])
    y = profile[order, 0].astype(float)
    z = profile[order, 1].astype(float)

    fit_degree = int(min(max(degree, 1), len(y) - 1))
    y_mean = float(np.mean(y))
    y_scale = float(0.5 * (np.max(y) - np.min(y)))
    if y_scale <= 1e-9:
        y_scale = 1.0

    y_normalized = (y - y_mean) / y_scale
    coefficients = np.polyfit(y_normalized, z, fit_degree)

    return {
        "coefficients": coefficients,
        "y_mean": y_mean,
        "y_scale": y_scale,
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "degree": float(fit_degree),
    }


def surface_z_at_y_fitted(
    surface_fit: dict[str, float | np.ndarray],
    y_values: np.ndarray | float,
) -> np.ndarray:
    """Evaluate the fitted material-surface curve Z=f(Y), allowing extrapolation."""
    scalar_input = np.isscalar(y_values)
    y_query = np.asarray([y_values], dtype=float) if scalar_input else np.asarray(y_values, dtype=float)

    y_mean = float(surface_fit["y_mean"])
    y_scale = float(surface_fit["y_scale"])
    coefficients = np.asarray(surface_fit["coefficients"], dtype=float)
    z_query = np.polyval(coefficients, (y_query - y_mean) / y_scale)

    outside = (y_query < float(surface_fit["y_min"])) | (y_query > float(surface_fit["y_max"]))
    if np.any(outside):
        print(
            "Warning: some trajectory Y values are outside the point-cloud surface range; "
            "fitted curve extrapolation was used."
        )

    return z_query[0] if scalar_input else z_query


def make_fitted_surface_curve_for_plot(
    surface_fit: dict[str, float | np.ndarray],
    trajectory_true: np.ndarray,
    trajectory_plan: np.ndarray,
    start_pos: np.ndarray,
    end_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the Y-Z curve shown in the 2D plot, with limited extrapolation."""
    y_ref = np.r_[
        trajectory_true[:, 0],
        trajectory_plan[:, 0],
        start_pos[0],
        end_pos[0],
    ]
    y_min_target = float(np.nanmin(y_ref) - FITTED_SURFACE_Y_MARGIN)
    y_max_target = float(np.nanmax(y_ref) + FITTED_SURFACE_Y_MARGIN)

    # Keep extrapolation bounded so a high-order polynomial cannot dominate the plot.
    y_min_allowed = float(surface_fit["y_min"]) - FITTED_SURFACE_EXTRAPOLATION_MM
    y_max_allowed = float(surface_fit["y_max"]) + FITTED_SURFACE_EXTRAPOLATION_MM
    y_curve_min = max(y_min_target, y_min_allowed)
    y_curve_max = min(y_max_target, y_max_allowed)

    if y_curve_max <= y_curve_min:
        y_curve_min = y_min_target
        y_curve_max = y_max_target

    y_curve = np.linspace(y_curve_min, y_curve_max, FITTED_SURFACE_NUM_POINTS)
    z_curve = surface_z_at_y_fitted(surface_fit, y_curve)
    return y_curve, z_curve

def calc_penetration_against_point_cloud_surface(
    trajectory_yz: np.ndarray,
    surface_profile: np.ndarray,
) -> np.ndarray:
    """
    Positive value means the bucket tip is below the point-cloud material surface.

    This is only used to find the excavation segment; it is not reported as an
    evaluation metric.
    """
    surface_z = surface_z_at_y(surface_profile, trajectory_yz[:, 0])
    return surface_z - trajectory_yz[:, 1]


def _extract_largest_motion_window(
    t: np.ndarray,
    distance: np.ndarray,
    trajectory_true: np.ndarray,
    duration: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Fallback extraction when no reliable material-surface contact is found."""
    dt = np.median(np.diff(t))
    window_len = max(3, int(round(duration / dt)))
    if window_len >= len(t):
        start_idx = 0
    else:
        step_dist = np.linalg.norm(np.diff(trajectory_true, axis=0), axis=1)
        cumsum = np.r_[0.0, np.cumsum(step_dist)]
        window_scores = cumsum[window_len:] - cumsum[:-window_len]
        start_idx = int(np.argmax(window_scores))

    end_time = t[start_idx] + duration
    end_idx = int(np.searchsorted(t, end_time, side="right"))
    end_idx = min(max(end_idx, start_idx + 2), len(t))

    return t[start_idx:end_idx] - t[start_idx], distance[start_idx:end_idx], trajectory_true[start_idx:end_idx], start_idx, end_idx


def extract_excavation_window(
    t: np.ndarray,
    distance: np.ndarray,
    trajectory_true: np.ndarray,
    duration: float,
    surface_profile: np.ndarray,
    start_time: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the excavation segment without polynomial material-surface fitting.

    If start_time is given, use [start_time, start_time + duration]. Otherwise,
    find the first bucket-tip penetration into the real point-cloud surface, then
    keep a duration-long segment. If no penetration is found, use the largest
    bucket-tip motion window as a fallback.
    """
    if duration <= 0:
        raise ValueError(f"Planned trajectory duration must be positive, got {duration}.")

    if start_time is not None:
        start_idx = int(np.searchsorted(t, start_time, side="left"))
        end_time = t[start_idx] + duration
        end_idx = int(np.searchsorted(t, end_time, side="right"))
        end_idx = min(max(end_idx, start_idx + 2), len(t))
        return t[start_idx:end_idx] - t[start_idx], distance[start_idx:end_idx], trajectory_true[start_idx:end_idx]

    penetration = calc_penetration_against_point_cloud_surface(trajectory_true, surface_profile)
    contact_indices = np.where(penetration > EXCAVATION_DEPTH_THRESHOLD_MM)[0]

    if contact_indices.size > 0:
        start_idx = int(contact_indices[0])
        end_time = t[start_idx] + duration
        end_idx = int(np.searchsorted(t, end_time, side="right"))
        end_idx = min(max(end_idx, start_idx + 2), len(t))
        return t[start_idx:end_idx] - t[start_idx], distance[start_idx:end_idx], trajectory_true[start_idx:end_idx]

    print("Warning: no bucket-tip penetration into the point-cloud surface was found; using largest-motion window.")
    t_ex, dis_ex, traj_ex, _, _ = _extract_largest_motion_window(t, distance, trajectory_true, duration)
    return t_ex, dis_ex, traj_ex


def define_control_para_no_surface_fit(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    end_angle: float,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Build spline control parameters without using a fitted material surface."""
    n_points = (len(x) - 2) // 2
    middle_pos = np.zeros((n_points, 3))
    for j in range(n_points):
        middle_pos[j, 0] = x[j * 2]
        middle_pos[j, 1] = x[j * 2 + 1]
        middle_pos[j, 2] = (x[-1] / (n_points + 1)) * (j + 1)

    start_angle = x[-2]
    t_end = x[-1]
    return start_pos, middle_pos, end_pos, start_angle, end_angle, t_end


def trajectory_planning_cubic_spline(
    start_pos: np.ndarray,
    middle_pos: np.ndarray,
    end_pos: np.ndarray,
    start_angle: float,
    end_angle: float,
    t_end: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Plan the bucket-tip trajectory using cubic splines."""
    control_points = np.array([start_pos, end_pos])
    t_control = np.array([0.0])

    for i in range(middle_pos.shape[0]):
        insert_pos = control_points.shape[0] - 1
        control_points = np.insert(control_points, insert_pos, middle_pos[i, :2], axis=0)
        t_control = np.append(t_control, middle_pos[i, 2] + t_control[i])
    t_control = np.append(t_control, t_end)

    control_angles = np.linspace(start_angle, end_angle, num=control_points.shape[0])
    cubic_spline_pos = CubicSpline(t_control, control_points, bc_type="clamped")
    cubic_spline_angle = CubicSpline(t_control, control_angles, bc_type="clamped")

    t = np.linspace(0, t_control[-1], num=100)
    trajectory = cubic_spline_pos(t)
    velocity = cubic_spline_pos(t, 1)
    acceleration = cubic_spline_pos(t, 2)
    tao = cubic_spline_angle(t)
    tao_v = cubic_spline_angle(t, 1)

    kne = Kinematic()
    dis = np.zeros((trajectory.shape[0], 3))
    pos = np.zeros((trajectory.shape[0], 6))
    for i in range(trajectory.shape[0]):
        dis[i, :] = kne.kine_wzToqd(0, trajectory[i, 0], trajectory[i, 1], tao[i])
        pos[i, :] = kne.kine_qdTowz(90, dis[i, 0], dis[i, 1], dis[i, 2])

    return pos, trajectory, velocity, acceleration, tao, tao_v, t, control_points, dis


def get_planning_traj_from_point_cloud_surface(
    trajectory_results: np.ndarray,
    trajectory_true: np.ndarray,
    surface_profile: np.ndarray,
    surface_fit: dict[str, float | np.ndarray] | None = None,
    end_angle: float = END_ANGLE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create the planned/mined trajectory with start/end snapped to the real point-cloud surface.

    This preserves the original idea: take start/end Y from the real excavation
    trajectory, but obtain start/end Z from the material surface. Unlike the
    original script, Z is taken from the real point cloud rather than from a
    polynomial fit.
    """
    start_y = float(trajectory_true[0, 0])
    end_y = float(trajectory_true[-1, 0])
    if USE_FITTED_SURFACE_FOR_ENDPOINTS and surface_fit is not None:
        start_z, end_z = surface_z_at_y_fitted(surface_fit, np.array([start_y, end_y]))
    else:
        start_z, end_z = surface_z_at_y(surface_profile, np.array([start_y, end_y]))

    start_pos = np.array([start_y, start_z])
    end_pos = np.array([end_y, end_z])

    start_pos, middle_pos, end_pos, start_angle, end_angle, t_end = define_control_para_no_surface_fit(
        start_pos, end_pos, end_angle, trajectory_results
    )
    _, trajectory, _, _, _, _, t, control_points, dis = trajectory_planning_cubic_spline(
        start_pos, middle_pos, end_pos, start_angle, end_angle, t_end
    )
    return t, dis, trajectory, start_pos, end_pos, control_points


def crop_point_cloud_for_trajectory(
    point_cloud: np.ndarray,
    trajectory_true: np.ndarray,
    trajectory_plan: np.ndarray | None = None,
    x_center: float = POINT_CLOUD_X_CENTER,
    x_margin: float = POINT_CLOUD_X_MARGIN,
    y_margin: float = POINT_CLOUD_Y_MARGIN,
    z_margin: float = POINT_CLOUD_Z_MARGIN,
    max_points: int = MAX_POINT_CLOUD_POINTS,
    random_seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Crop transformed 3D point cloud to the region around the compared trajectories.

    If the input contains RGB columns, the cropped output keeps them as Nx6.
    """
    points, colors = _split_xyz_rgb(point_cloud)

    traj = trajectory_true if trajectory_plan is None else np.vstack([trajectory_true, trajectory_plan])
    y_min, z_min = np.min(traj, axis=0)
    y_max, z_max = np.max(traj, axis=0)

    mask = (
        (points[:, 0] >= x_center - x_margin)
        & (points[:, 0] <= x_center + x_margin)
        & (points[:, 1] >= y_min - y_margin)
        & (points[:, 1] <= y_max + y_margin)
        & (points[:, 2] >= z_min - z_margin)
        & (points[:, 2] <= z_max + z_margin)
    )
    cropped_points = points[mask]
    cropped_colors = colors[mask] if colors is not None else None

    if cropped_points.shape[0] > max_points:
        rng = np.random.default_rng(random_seed)
        keep = rng.choice(cropped_points.shape[0], size=max_points, replace=False)
        cropped_points = cropped_points[keep]
        cropped_colors = cropped_colors[keep] if cropped_colors is not None else None

    return _combine_xyz_rgb(cropped_points, cropped_colors)


def set_plot_style() -> None:
    config = {
        "font.family": "serif",
        "font.size": 8,
        "mathtext.fontset": "stix",
        "font.serif": ["Times New Roman"],
    }
    rcParams.update(config)


def plot_trajectory(
    trajectory_true: np.ndarray,
    trajectory_plan: np.ndarray,
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    control_points: np.ndarray,
    point_cloud: np.ndarray,
    surface_fit: dict[str, float | np.ndarray] | None = None,
) -> np.ndarray:
    """
    Plot trajectory comparison in the 2D Y-Z plane.

    The point cloud is still cropped and returned for the optional 3D view, but
    it is not displayed in the 2D figure when DRAW_POINT_CLOUD_2D=False. When
    DRAW_FITTED_SURFACE_2D=True, a fitted/extrapolated material-surface curve is
    drawn in this 2D view.
    """
    set_plot_style()
    cropped_points = crop_point_cloud_for_trajectory(point_cloud, trajectory_true, trajectory_plan)

    plt.figure(figsize=(FIGURE_2D_SIZE_CM[0] / 2.54, FIGURE_2D_SIZE_CM[1] / 2.54))

    if DRAW_POINT_CLOUD_2D:
        if cropped_points.size > 0:
            _, cropped_colors = _split_xyz_rgb(cropped_points)
            scatter_kwargs = {
                "s": 0.7,
                "alpha": POINT_CLOUD_ALPHA_2D,
                "linewidths": 0,
                "label": "Material point cloud",
                "zorder": 1,
            }
            if cropped_colors is not None:
                scatter_kwargs["c"] = cropped_colors
            else:
                scatter_kwargs["c"] = "0.72"
            plt.scatter(cropped_points[:, 1], cropped_points[:, 2], **scatter_kwargs)
        else:
            print("Warning: no point-cloud points remain after cropping. Check crop margins.")

    if DRAW_FITTED_SURFACE_2D and surface_fit is not None:
        y_fit, z_fit = make_fitted_surface_curve_for_plot(
            surface_fit, trajectory_true, trajectory_plan, start_pos, end_pos
        )
        plt.plot(
            y_fit,
            z_fit,
            "-.",
            color="0.35",
            label="Fitted material surface",
            linewidth=FITTED_SURFACE_LINEWIDTH,
            alpha=0.85,
            zorder=2,
        )

    plt.plot(trajectory_plan[:, 0], trajectory_plan[:, 1], "--", label="Trajectory plan", linewidth=1.45, zorder=4)
    plt.plot(trajectory_true[:, 0], trajectory_true[:, 1], "-", label="Trajectory true", linewidth=1.45, zorder=5)
    plt.scatter(control_points[:, 0], control_points[:, 1], color="red", label="Control Points", s=9, zorder=6)
    plt.scatter(start_pos[0], start_pos[1], color="green", label="Start Point", s=14, zorder=7)
    plt.scatter(end_pos[0], end_pos[1], color="blue", label="End Point", s=14, zorder=7)

    plt.xlabel("Y (mm)")
    plt.ylabel("Z (mm)")
    if LEGEND_OUTSIDE_2D:
        plt.legend(
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=LEGEND_NCOL_2D,
            borderaxespad=0.0,
            handlelength=1.8,
            columnspacing=1.0,
        )
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))
    else:
        plt.legend(frameon=True, framealpha=0.65, loc="best", ncol=2)
        plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "Compare_trajectory_pointcloud.svg", format="svg", bbox_inches="tight")
    plt.show()

    return cropped_points



def _set_3d_limits_and_aspect(
    ax,
    point_cloud_xyz: np.ndarray,
    trajectory_true: np.ndarray,
    trajectory_plan: np.ndarray,
    control_points: np.ndarray,
) -> None:
    """Set 3D axis limits and box aspect for clearer visual proportions."""
    trajectory_true_xyz = np.c_[np.zeros(trajectory_true.shape[0]), trajectory_true]
    trajectory_plan_xyz = np.c_[np.zeros(trajectory_plan.shape[0]), trajectory_plan]
    control_points_xyz = np.c_[np.zeros(control_points.shape[0]), control_points]
    all_xyz = np.vstack([point_cloud_xyz, trajectory_true_xyz, trajectory_plan_xyz, control_points_xyz])

    xyz_min = np.nanmin(all_xyz, axis=0)
    xyz_max = np.nanmax(all_xyz, axis=0)
    xyz_range = xyz_max - xyz_min
    xyz_range[xyz_range <= 1e-9] = 1.0

    margin = xyz_range * LIMITS_3D_MARGIN_RATIO
    ax.set_xlim(xyz_min[0] - margin[0], xyz_max[0] + margin[0])
    ax.set_ylim(xyz_min[1] - margin[1], xyz_max[1] + margin[1])
    ax.set_zlim(xyz_min[2] - margin[2], xyz_max[2] + margin[2])

    try:
        if USE_DATA_ASPECT_3D:
            ax.set_box_aspect(tuple(xyz_range))
        else:
            ax.set_box_aspect(MANUAL_BOX_ASPECT_3D)
    except AttributeError:
        print(
            "Warning: current Matplotlib does not support ax.set_box_aspect; "
            "upgrade Matplotlib if you need exact 3D display aspect control."
        )



def _format_3d_axes(ax) -> None:
    """Reduce 3D tick-label overlap and keep all requested axis ticks visible."""
    # Important: do NOT reuse the same Locator/Formatter instance across axes.
    # Matplotlib locators keep a reference to their owning axis, so sharing one
    # locator may make one axis, commonly Y in 3D views, lose its tick labels.
    axis_items = (
        (ax.xaxis, HIDE_X_TICK_LABELS_3D),
        (ax.yaxis, HIDE_Y_TICK_LABELS_3D),
        (ax.zaxis, HIDE_Z_TICK_LABELS_3D),
    )

    for axis, hide_labels in axis_items:
        axis.set_major_locator(MaxNLocator(nbins=MAX_TICKS_3D))
        axis.set_major_formatter(FormatStrFormatter(TICK_FORMAT_3D))
        axis.set_tick_params(labelsize=TICK_LABEL_SIZE_3D, pad=TICK_PAD_3D)
        for tick_label in axis.get_ticklabels():
            tick_label.set_visible(not hide_labels)

    ax.xaxis.label.set_size(AXIS_LABEL_SIZE_3D)
    ax.yaxis.label.set_size(AXIS_LABEL_SIZE_3D)
    ax.zaxis.label.set_size(AXIS_LABEL_SIZE_3D)

    ax.xaxis.labelpad = AXIS_LABEL_PAD_3D
    ax.yaxis.labelpad = AXIS_LABEL_PAD_3D
    ax.zaxis.labelpad = AXIS_LABEL_PAD_3D


# def _set_3d_margins(fig, ax) -> None:
#     """Make the 3D axes occupy most of the canvas."""
#     ax.set_position(AXES_POSITION_3D)
#     fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

def _set_3d_margins(fig, ax) -> None:
    """Make the 3D axes occupy most of the canvas, while keeping Z label visible."""
    # fig.subplots_adjust(left=0.02, right=0.95, bottom=0.02, top=0.7)
    ax.set_position(AXES_POSITION_3D)

def plot_trajectory_3d(
    cropped_points: np.ndarray,
    trajectory_true: np.ndarray,
    trajectory_plan: np.ndarray,
    control_points: np.ndarray,
) -> None:
    """Optional 3D view of the cropped point cloud and the two trajectories."""
    if cropped_points.size == 0:
        return

    set_plot_style()
    fig = plt.figure(figsize=(FIGURE_3D_SIZE_CM[0] / 2.54, FIGURE_3D_SIZE_CM[1] / 2.54))
    ax = fig.add_subplot(111, projection="3d")

    _, cropped_colors = _split_xyz_rgb(cropped_points)
    scatter_kwargs = {
        "s": 0.7,
        "alpha": POINT_CLOUD_ALPHA_3D,
        "label": "Material point cloud",
    }
    if cropped_colors is not None:
        scatter_kwargs["c"] = cropped_colors
    else:
        scatter_kwargs["c"] = "0.72"

    ax.scatter(
        cropped_points[:, 0],
        cropped_points[:, 1],
        cropped_points[:, 2],
        **scatter_kwargs,
    )

    _set_3d_limits_and_aspect(ax, cropped_points[:, :3], trajectory_true, trajectory_plan, control_points)
    ax.view_init(elev=VIEW_3D_ELEV, azim=VIEW_3D_AZIM)

    x0_true = np.zeros(trajectory_true.shape[0])
    x0_plan = np.zeros(trajectory_plan.shape[0])
    x0_ctrl = np.zeros(control_points.shape[0])
    ax.plot(x0_plan, trajectory_plan[:, 0], trajectory_plan[:, 1], "--", linewidth=1, label="Trajectory plan")
    ax.plot(x0_true, trajectory_true[:, 0], trajectory_true[:, 1], "-", linewidth=1, label="Trajectory true")
    ax.scatter(x0_ctrl, control_points[:, 0], control_points[:, 1], color="red", s=5, label="Control Points")

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    _format_3d_axes(ax)
    # ax.legend(frameon=False, 
    #           loc="upper center", 
    #           ncol=2, 
    #           fontsize=TICK_LABEL_SIZE_3D, 
    #           handlelength=1., 
    #           borderaxespad=0.2)
    # _set_3d_margins(fig, ax)
    _set_3d_margins(fig, ax)

    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.9),   # 数值越小，图例越靠近三维图
        bbox_transform=fig.transFigure,
        ncol=2,
        fontsize=TICK_LABEL_SIZE_3D,
        handlelength=1.2,
        columnspacing=0.8,
        borderaxespad=0.0,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "Compare_trajectory_pointcloud_3d.svg", format="svg", bbox_inches="tight", pad_inches=SAVE_PAD_INCHES_3D)
    # plt.savefig(
    # OUTPUT_DIR / "Compare_trajectory_pointcloud_3d.svg",
    # format="svg",
    # bbox_inches="tight",
    # pad_inches=0.0,
    # )
    plt.show()



def save_trajectory_3d_view_candidates(
    cropped_points: np.ndarray,
    trajectory_true: np.ndarray,
    trajectory_plan: np.ndarray,
    control_points: np.ndarray,
) -> None:
    """Save several candidate 3D views to help choose elev/azim values."""
    if cropped_points.size == 0:
        return

    for elev, azim in CANDIDATE_3D_VIEWS:
        set_plot_style()
        fig = plt.figure(figsize=(FIGURE_3D_SIZE_CM[0] / 2.54, FIGURE_3D_SIZE_CM[1] / 2.54))
        ax = fig.add_subplot(111, projection="3d")

        _, cropped_colors = _split_xyz_rgb(cropped_points)
        scatter_kwargs = {
            "s": 0.7,
            "alpha": POINT_CLOUD_ALPHA_3D,
            "label": "Material point cloud",
        }
        if cropped_colors is not None:
            scatter_kwargs["c"] = cropped_colors
        else:
            scatter_kwargs["c"] = "0.72"

        ax.scatter(cropped_points[:, 0], cropped_points[:, 1], cropped_points[:, 2], **scatter_kwargs)
        _set_3d_limits_and_aspect(ax, cropped_points[:, :3], trajectory_true, trajectory_plan, control_points)
        ax.view_init(elev=elev, azim=azim)

        x0_true = np.zeros(trajectory_true.shape[0])
        x0_plan = np.zeros(trajectory_plan.shape[0])
        x0_ctrl = np.zeros(control_points.shape[0])
        ax.plot(x0_plan, trajectory_plan[:, 0], trajectory_plan[:, 1], "--", linewidth=1, label="Trajectory plan")
        ax.plot(x0_true, trajectory_true[:, 0], trajectory_true[:, 1], "-", linewidth=1, label="Trajectory true")
        ax.scatter(x0_ctrl, control_points[:, 0], control_points[:, 1], color="red", s=5, label="Control Points")

        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        _format_3d_axes(ax)
        ax.legend(frameon=False, loc="upper left", fontsize=TICK_LABEL_SIZE_3D, handlelength=1.4, borderaxespad=0.2)
        _set_3d_margins(fig, ax)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_file = OUTPUT_DIR / f"Compare_trajectory_pointcloud_3d_elev{elev}_azim{azim}.svg"
        plt.savefig(out_file, format="svg", bbox_inches="tight", pad_inches=SAVE_PAD_INCHES_3D)
        plt.close(fig)
        print(f"Saved candidate 3D view: {out_file}")


def main() -> None:
    sensor_file = DATA_DIR / "CX_test" / f"sensor_data_{FILE_INDEX}.csv"
    point_cloud_file = DATA_DIR / "Point_cloud_material_face" / f"Point_cloud_material_face_{FILE_INDEX}.ply"
    point_pose_file = DATA_DIR / "Point_cloud_material_face_pos" / f"Point_cloud_material_face_pos_{FILE_INDEX}.txt"
    trajectory_results_file = DATA_DIR / "Trajectory_planning_results" / f"Trajectory_planning_results_{FILE_INDEX}.npy"

    # 1) Read sensor displacement only; do not calculate resistance.
    t_true_all, dis_true_all = data_filter_distance_only(sensor_file)
    trajectory_true_all = calc_true_trajectory(dis_true_all)

    # 2) Load and transform the real point cloud first, because the point-cloud
    #    surface is used to locate excavation start and to place planning endpoints.
    point_cloud = get_transformed_point_cloud(point_cloud_file, point_pose_file)
    surface_profile = build_surface_profile_from_point_cloud(point_cloud)

    # 3) Read planned trajectory parameters. The last value is the planned duration.
    trajectory_results = np.load(trajectory_results_file).reshape(-1)[0:4]
    planned_duration = float(trajectory_results[-1])

    # 4) Extract the true excavation trajectory using the real point-cloud surface,
    #    without polynomial fitting and without calculating resistance/volume metrics.
    #    This extraction still uses the full observed surface profile so the window
    #    can be found robustly. The fitted display curve below is local only.
    t_true, dis_true, trajectory_true = extract_excavation_window(
        t_true_all,
        dis_true_all,
        trajectory_true_all,
        duration=planned_duration,
        surface_profile=surface_profile,
        start_time=EXCAVATION_START_TIME,
    )

    # 5) Fit a 4th-order Y-Z material-surface curve using only point-cloud surface
    #    samples near the extracted excavation trajectory, then allow bounded
    #    extrapolation for the missing endpoint side.
    if DRAW_FITTED_SURFACE_2D or USE_FITTED_SURFACE_FOR_ENDPOINTS:
        local_surface_profile = select_surface_profile_near_trajectory(surface_profile, trajectory_true)
        surface_fit = fit_material_surface_curve(local_surface_profile, degree=FITTED_SURFACE_DEGREE)
    else:
        surface_fit = None

    # 6) Generate the planned/mined trajectory. Its start/end Y come from the real
    #    excavation segment, and its start/end Z are snapped to the local fitted
    #    point-cloud surface curve when enabled.
    t_plan, dis_plan, trajectory_plan, start_pos_plan, end_pos_plan, control_points = (
        get_planning_traj_from_point_cloud_surface(
            trajectory_results,
            trajectory_true,
            surface_profile,
            surface_fit=surface_fit,
        )
    )

    # 6) Plot only the cropped real point cloud near the compared trajectories.
    cropped_points = plot_trajectory(
        trajectory_true,
        trajectory_plan,
        start_pos_plan,
        end_pos_plan,
        control_points,
        point_cloud,
        surface_fit=surface_fit,
    )

    if SAVE_3D_VIEW:
        plot_trajectory_3d(cropped_points, trajectory_true, trajectory_plan, control_points)
        if SAVE_3D_VIEW_CANDIDATES:
            save_trajectory_3d_view_candidates(cropped_points, trajectory_true, trajectory_plan, control_points)

    print(f"Excavation samples: {len(t_true)}")
    print(f"Cropped point-cloud points for 3D view: {len(cropped_points)}")
    print(f"2D point cloud displayed: {DRAW_POINT_CLOUD_2D}")
    print(f"Point-cloud RGB kept: {_split_xyz_rgb(cropped_points)[1] is not None}")
    print(f"Plan start point snapped to point-cloud surface: {start_pos_plan}")
    print(f"Plan end point snapped to point-cloud surface: {end_pos_plan}")
    if surface_fit is not None:
        print(f"2D fitted surface degree: {int(surface_fit['degree'])}")
        if USE_FITTED_SURFACE_FOR_ENDPOINTS:
            print("Planning endpoints Z: fitted/extrapolated surface curve")
    print(f"Saved: {OUTPUT_DIR / 'Compare_trajectory_pointcloud.svg'}")
    if SAVE_3D_VIEW:
        print(f"Saved: {OUTPUT_DIR / 'Compare_trajectory_pointcloud_3d.svg'}")
        print(f"3D view angle: elev={VIEW_3D_ELEV}, azim={VIEW_3D_AZIM}")
        print(f"3D max ticks per axis: {MAX_TICKS_3D}")
        print(f"3D tick labels hidden: X={HIDE_X_TICK_LABELS_3D}, Y={HIDE_Y_TICK_LABELS_3D}, Z={HIDE_Z_TICK_LABELS_3D}")
        print(f"3D axes position: {AXES_POSITION_3D}, save pad inches: {SAVE_PAD_INCHES_3D}")
        if USE_DATA_ASPECT_3D:
            print("3D box aspect: data range of cropped cloud + trajectories")
        else:
            print(f"3D box aspect X:Y:Z = {MANUAL_BOX_ASPECT_3D}")


if __name__ == "__main__":
    main()
