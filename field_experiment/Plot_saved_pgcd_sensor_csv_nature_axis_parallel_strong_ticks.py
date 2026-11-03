from __future__ import annotations

"""
Read a saved PGCD planned sensor-like CSV and redraw the trajectory figure.

Input planned CSV format:
  column 0    : time
  columns 1-3 : planned cylinder displacements
  columns 4+  : optional unused columns, kept for sensor_data_*.csv compatibility

This script does not run PGCD again. It only reads the saved CSV, reconstructs
the bucket-tip Y-Z trajectory from the three displacement columns, loads the
current material point cloud, and draws a combined 2D + 3D figure.

This version is prepared for SCI/Nature-style figure output with:
1. Nature-style Arial/Helvetica typography.
2. Separate legends for panels B and C.
3. All requested legend entries retained.
4. 2D and 3D panel boxes share the same vertical position.
5. The 3D point cloud keeps its original RGB colors when available.
6. Point-cloud down-sampling is disabled by default and can be controlled by parameter.
7. 3D axis labels are projected and rotated parallel to the displayed axis directions.
"""

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from mpl_toolkits.mplot3d import proj3d


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
PLATFORM_ROOT = PROJECT_ROOT
for _path in (PLATFORM_ROOT, PROJECT_ROOT, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

DATA_DIR = PROJECT_ROOT / "experiment_data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "pgcd_current_material_outputs"
DEFAULT_FILE_INDEX = 5

# =========================
# Figure layout controls
# =========================
FIGURE_SIZE_CM = (10, 5)
DPI = 600

# Explicit axes positions in figure coordinates: [left, bottom, width, height].
# The 2D and 3D panels share the same bottom/top coordinates, so their
# plotting areas are horizontally aligned instead of being vertically staggered.
PANEL_BOTTOM_2D = 0.150
PANEL_HEIGHT_2D = 0.45
PANEL_WIDTH_2D = 0.35

PANEL_BOTTOM_3D = 0.0150
PANEL_HEIGHT_3D = 0.7
PANEL_WIDTH_3D = 0.7

PANEL_TOP = 0.85
AX2D_POSITION = (0.11, PANEL_BOTTOM_2D, PANEL_WIDTH_2D, PANEL_HEIGHT_2D)
AX3D_POSITION = (0.400, PANEL_BOTTOM_3D, PANEL_WIDTH_3D, PANEL_HEIGHT_3D)

# Do not call ax2d.set_box_aspect here. It silently shrinks/re-centres the
# 2D axes inside AX2D_POSITION and makes the two panels look misaligned.
# To change the 2D panel aspect, adjust AX2D_POSITION width/height instead.

# Separate legend and panel-label positions in figure coordinates.
LEGEND_FIG_Y = 0.8
LEGEND2D_FIG_X = AX2D_POSITION[0] + 0.5 * AX2D_POSITION[2]
LEGEND3D_FIG_X = AX3D_POSITION[0] + 0.5 * AX3D_POSITION[2]
PANEL_LABEL_FIG_Y = PANEL_TOP + 0.018
PANEL_LABEL_X_OFFSET = 0.020
LEGEND2D_NCOL = 2
LEGEND3D_NCOL = 2

# 3D internal visual aspect X:Y:Z.
USE_DATA_ASPECT_3D = False
MANUAL_BOX_ASPECT_3D = (1.15, 1.45, 1.05)
LIMITS_3D_MARGIN_RATIO = 0.015

# Projected 3D axis labels. The labels are drawn in 2D after the 3D view is set,
# and their rotation is calculated from the screen projection of each 3D axis.
# This makes X/Y/Z labels visually parallel to the displayed 3D axes.
USE_PROJECTED_AXIS_PARALLEL_LABELS_3D = True
AXIS_PARALLEL_LABEL_OFFSET_3D = 0.18
X3D_LABEL_T = 0.50
Y3D_LABEL_T = 0.58
Z3D_LABEL_T = 0.52

# =========================
# Typography and line controls
# =========================
FONT_SIZE_PT = 6
PANEL_LABEL_SIZE_PT = 6
LEGEND_FONT_SIZE_PT = 6
AXIS_LINEWIDTH = 0.55
TICK_WIDTH = 0.55
TICK_LENGTH = 2.2

# 3D tick/axis visibility controls. These are intentionally separate from
# the 2D axes, because 3D ticks become faint after projection and resizing.
TICK_LABEL_SIZE_3D = 7.0
TICK_PAD_3D = 0
TICK_COLOR_3D = "0.05"
AXIS_LINEWIDTH_3D = 0.85
TICK_LINEWIDTH_3D = 0.75
GRID_LINEWIDTH_3D = 0.50
GRID_COLOR_3D = (0.55, 0.55, 0.55, 0.78)
PANE_EDGE_COLOR_3D = (0.55, 0.55, 0.55, 0.90)

SURFACE_LINEWIDTH = 0.75
TRAJECTORY_LINEWIDTH_2D = 0.90
TRAJECTORY_LINEWIDTH_3D = 0.85

POINT_CLOUD_ALPHA_3D = 0.16
POINT_CLOUD_POINT_SIZE_3D = 0.8
CONTROL_POINT_SIZE_2D = 8
CONTROL_POINT_SIZE_3D = 6
ENDPOINT_POINT_SIZE_2D = 13
ENDPOINT_POINT_SIZE_3D = 10

# Keep the material-face point cloud in its original RGB color when available.
USE_POINT_CLOUD_RGB_3D = True

# 0 or a negative value means no random down-sampling after cropping.
# A positive value limits the plotted cropped point-cloud rows to that number.
MAX_POINT_CLOUD_POINTS_3D = 8000
POINT_CLOUD_RANDOM_SEED_3D = 0

# Color-blind friendly palette.
COLOR_SURFACE = "0.35"
COLOR_CLOUD = "0.68"
COLOR_PLAN = "#D55E00"
COLOR_MEASURED = "#0072B2"
COLOR_CONTROL = "#CC79A7"
COLOR_START = "#009E73"
COLOR_END = "#000000"

LABEL_MATERIAL_SURFACE = "Material surface"
LABEL_MATERIAL_FACE = "Material point cloud"
LABEL_PLANNED_TRAJECTORY = "Planned trajectory"
LABEL_MEASURED_TRAJECTORY = "Measured trajectory"
LABEL_CONTROL_POINTS = "Control points"
LABEL_START_POINT = "Start point"
LABEL_END_POINT = "End point"


def load_viz_module(viz_script: Path | None = None) -> ModuleType:
    """Load Compare_trajectory_pointcloud_v9/v8 helper functions."""
    if viz_script is not None:
        viz_script = Path(viz_script)
        if not viz_script.exists():
            raise FileNotFoundError(f"VIZ script not found: {viz_script}")
        spec = importlib.util.spec_from_file_location("trajectory_viz_helpers", viz_script)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import VIZ script: {viz_script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    for module_name in ("Compare_trajectory_pointcloud_v9", "Compare_trajectory_pointcloud_v8"):
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            pass

    for file_name in (
        "Compare_trajectory_pointcloud_v9.py",
        "Compare_trajectory_pointcloud_v9(2).py",
        "Compare_trajectory_pointcloud_v8.py",
    ):
        candidate = SCRIPT_DIR / file_name
        if candidate.exists():
            return load_viz_module(candidate)

    raise ModuleNotFoundError(
        "Could not find Compare_trajectory_pointcloud_v9.py or Compare_trajectory_pointcloud_v8.py. "
        "Pass --viz-script to the helper script explicitly."
    )


def default_planned_csv(file_index: int, output_dir: Path) -> Path:
    return output_dir / f"PGCD_planned_sensor_data_{file_index}.csv"


def sensor_file_for_index(file_index: int) -> Path:
    candidates = [
        DATA_DIR / "CX_test" / f"sensor_data_{file_index}.csv",
        DATA_DIR / "Data" / "CX_test" / f"sensor_data_{file_index}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Actual sensor_data file not found for index {file_index}: {candidates[0]}")


def point_cloud_files_for_index(file_index: int) -> tuple[Path, Path]:
    point_cloud_candidates = [
        DATA_DIR / "Point_cloud_material_face" / f"Point_cloud_material_face_{file_index}.ply",
        DATA_DIR / f"Point_cloud_material_face_{file_index}.ply",
    ]
    pose_candidates = [
        DATA_DIR / "Point_cloud_material_face_pos" / f"Point_cloud_material_face_pos_{file_index}.txt",
        DATA_DIR / f"Point_cloud_material_face_pos_{file_index}.txt",
    ]

    point_cloud_file = next((p for p in point_cloud_candidates if p.exists()), None)
    point_pose_file = next((p for p in pose_candidates if p.exists()), None)

    if point_cloud_file is None:
        raise FileNotFoundError(f"Point-cloud file not found for index {file_index}: {point_cloud_candidates[0]}")
    if point_pose_file is None:
        raise FileNotFoundError(f"Point-cloud pose file not found for index {file_index}: {pose_candidates[0]}")
    return point_cloud_file, point_pose_file


def read_sensor_like_csv(csv_file: Path, skip_rows: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Read time and three displacement columns from a sensor-like CSV."""
    csv_file = Path(csv_file)
    if not csv_file.exists():
        raise FileNotFoundError(f"Sensor-like CSV not found: {csv_file}")

    data = np.loadtxt(csv_file, delimiter=",", skiprows=skip_rows, usecols=[0, 1, 2, 3])
    data = np.atleast_2d(data)
    if data.shape[0] < 2:
        raise ValueError(f"Need at least two rows in {csv_file}, got {data.shape[0]}.")

    t = data[:, 0].astype(float)
    t = t - t[0]
    dis = data[:, 1:4].astype(float)
    return t, dis


def load_optional_control_points(control_points_csv: Path | None, planned_csv: Path) -> np.ndarray | None:
    candidates: list[Path] = []
    if control_points_csv is not None:
        candidates.append(Path(control_points_csv))
    candidates.append(planned_csv.with_name(f"{planned_csv.stem}_control_points_yz.csv"))

    for path in candidates:
        if path.exists():
            try:
                return np.loadtxt(path, delimiter=",", skiprows=1)
            except ValueError:
                return np.loadtxt(path, delimiter=",")
    return None


def fit_surface_for_plot(VIZ: ModuleType, point_cloud: np.ndarray, reference_trajectory: np.ndarray):
    surface_profile = VIZ.build_surface_profile_from_point_cloud(point_cloud)
    local_surface_profile = VIZ.select_surface_profile_near_trajectory(surface_profile, reference_trajectory)
    return VIZ.fit_material_surface_curve(local_surface_profile)


def surface_z_at_y_fitted_silent(surface_fit: dict[str, float | np.ndarray], y_values: np.ndarray | float) -> np.ndarray:
    scalar_input = np.isscalar(y_values)
    y_query = np.asarray([y_values], dtype=float) if scalar_input else np.asarray(y_values, dtype=float)
    y_mean = float(surface_fit["y_mean"])
    y_scale = float(surface_fit["y_scale"])
    coefficients = np.asarray(surface_fit["coefficients"], dtype=float)
    z_query = np.polyval(coefficients, (y_query - y_mean) / y_scale)
    return z_query[0] if scalar_input else z_query


def make_surface_curve(surface_fit: dict[str, float | np.ndarray], *trajectories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_all = np.concatenate([traj[:, 0] for traj in trajectories if traj is not None and len(traj) > 0])
    y_min = float(np.nanmin(y_all) - 90.0)
    y_max = float(np.nanmax(y_all) + 90.0)
    y_fit_min = float(surface_fit.get("y_min", y_min)) - 250.0
    y_fit_max = float(surface_fit.get("y_max", y_max)) + 250.0
    y_curve = np.linspace(max(y_min, y_fit_min), min(y_max, y_fit_max), 400)
    z_curve = surface_z_at_y_fitted_silent(surface_fit, y_curve)
    return y_curve, z_curve


def extract_actual_trajectory_for_duration(
    VIZ: ModuleType,
    actual_sensor_csv: Path,
    point_cloud: np.ndarray,
    planned_duration: float,
    actual_skip_rows: int,
) -> np.ndarray:
    t_actual_all, dis_actual_all = read_sensor_like_csv(actual_sensor_csv, skip_rows=actual_skip_rows)
    actual_trajectory_all = VIZ.calc_true_trajectory(dis_actual_all)
    surface_profile = VIZ.build_surface_profile_from_point_cloud(point_cloud)
    _, _, actual_trajectory = VIZ.extract_excavation_window(
        t=t_actual_all,
        distance=dis_actual_all,
        trajectory_true=actual_trajectory_all,
        duration=float(planned_duration),
        surface_profile=surface_profile,
        start_time=getattr(VIZ, "EXCAVATION_START_TIME", None),
    )
    return actual_trajectory


def apply_nature_style() -> None:
    """Apply compact Nature-style typography with Times New Roman."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": FONT_SIZE_PT,
            "axes.labelsize": FONT_SIZE_PT,
            "axes.titlesize": FONT_SIZE_PT,
            "legend.fontsize": LEGEND_FONT_SIZE_PT,
            "xtick.labelsize": FONT_SIZE_PT,
            "ytick.labelsize": FONT_SIZE_PT,
            "axes.linewidth": AXIS_LINEWIDTH,
            "xtick.major.width": TICK_WIDTH,
            "ytick.major.width": TICK_WIDTH,
            "xtick.major.size": TICK_LENGTH,
            "ytick.major.size": TICK_LENGTH,
            "figure.dpi": 600,
            "savefig.dpi": DPI,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_2d_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(AXIS_LINEWIDTH)
    ax.spines["bottom"].set_linewidth(AXIS_LINEWIDTH)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE_PT, width=TICK_WIDTH, length=TICK_LENGTH)
    ax.grid(False)


def format_3d_axes(ax) -> None:
    """Format 3D axes for a compact publication figure with visible ticks."""
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        axis.set_major_formatter(FormatStrFormatter("%.0f"))
        axis.set_tick_params(
            labelsize=TICK_LABEL_SIZE_3D,
            pad=TICK_PAD_3D,
            colors=TICK_COLOR_3D,
            width=TICK_LINEWIDTH_3D,
            length=3.0,
        )
        axis.label.set_size(FONT_SIZE_PT)

        # Make 3D grid, axis lines and tick marks readable after the figure is
        # reduced to journal-column size. The _axinfo fields are Matplotlib's
        # internal 3D styling interface, so each edit is protected by try/except.
        try:
            axis._axinfo["grid"].update(
                {
                    "linewidth": GRID_LINEWIDTH_3D,
                    "linestyle": "-",
                    "color": GRID_COLOR_3D,
                }
            )
            axis._axinfo["axisline"].update(
                {
                    "linewidth": AXIS_LINEWIDTH_3D,
                    "color": (0.12, 0.12, 0.12, 1.0),
                }
            )
            axis._axinfo["tick"].update(
                {
                    "inward_factor": 0.0,
                    "outward_factor": 0.30,
                    "linewidth": {True: TICK_LINEWIDTH_3D, False: 0.55},
                    "color": TICK_COLOR_3D,
                }
            )
        except Exception:
            pass

        try:
            axis.line.set_color(TICK_COLOR_3D)
            axis.line.set_linewidth(AXIS_LINEWIDTH_3D)
        except Exception:
            pass

        for tick_label in axis.get_ticklabels():
            tick_label.set_color(TICK_COLOR_3D)
            tick_label.set_fontsize(TICK_LABEL_SIZE_3D)

    for pane_axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            pane_axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            pane_axis.pane.set_edgecolor(PANE_EDGE_COLOR_3D)
            pane_axis.pane.set_linewidth(AXIS_LINEWIDTH_3D)
        except Exception:
            pass

    ax.xaxis.labelpad = -1
    ax.yaxis.labelpad = -1
    ax.zaxis.labelpad = -8

    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass


def _project_3d_point_to_axes(ax, point: tuple[float, float, float]) -> np.ndarray:
    """Project a 3D point to the 2D axes coordinate system."""
    x2, y2, _ = proj3d.proj_transform(point[0], point[1], point[2], ax.get_proj())
    x_display, y_display = ax.transData.transform((x2, y2))
    return ax.transAxes.inverted().transform((x_display, y_display))


def _screen_angle_between_3d_points(
    ax,
    point_start: tuple[float, float, float],
    point_end: tuple[float, float, float],
) -> float:
    """Return the screen-space angle from point_start to point_end in degrees."""
    p0_axes = _project_3d_point_to_axes(ax, point_start)
    p1_axes = _project_3d_point_to_axes(ax, point_end)
    p0_display = ax.transAxes.transform(p0_axes)
    p1_display = ax.transAxes.transform(p1_axes)
    dx, dy = p1_display - p0_display
    angle = float(np.degrees(np.arctan2(dy, dx)))

    # Keep labels readable from left to right when the projected axis is reversed.
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    return angle


def _axis_parallel_label_position(
    ax,
    point_start: tuple[float, float, float],
    point_end: tuple[float, float, float],
    t: float,
    offset: float,
) -> tuple[np.ndarray, float]:
    """Return a projected label position and rotation parallel to a 3D axis edge."""
    p0 = _project_3d_point_to_axes(ax, point_start)
    p1 = _project_3d_point_to_axes(ax, point_end)
    pos = (1.0 - t) * p0 + t * p1

    v = p1 - p0
    norm = float(np.linalg.norm(v))
    if norm > 1e-12:
        normal = np.array([-v[1], v[0]], dtype=float) / norm
        # Push the label away from the centre of the 3D axes box.
        centre = np.array([0.5, 0.5], dtype=float)
        if float(np.dot(normal, pos - centre)) < 0.0:
            normal = -normal
        pos = pos + offset * normal

    angle = _screen_angle_between_3d_points(ax, point_start, point_end)
    return pos, angle


def add_projected_axis_parallel_3d_labels(ax) -> None:
    """Draw X/Y/Z labels parallel to their projected 3D axes.

    Matplotlib's native 3D axis labels are not positioned like ordinary 2D labels:
    their distance and orientation depend on the current 3D projection. This helper
    hides the native titles and uses the projected axis directions instead.
    """
    if not USE_PROJECTED_AXIS_PARALLEL_LABELS_3D:
        return

    # Ensure the projection matrix and transforms are up to date.
    ax.figure.canvas.draw()

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    # Axis edges chosen to match the visible lower/front edges for the current view.
    # If the view angle is changed strongly, only these three edge definitions may
    # need to be adjusted.
    x_edge = ((xlim[0], ylim[0], zlim[0]), (xlim[1], ylim[0], zlim[0]))
    y_edge = ((xlim[1], ylim[0], zlim[0]), (xlim[1], ylim[1], zlim[0]))
    z_edge = ((xlim[1], ylim[1], zlim[0]), (xlim[1], ylim[1], zlim[1]))

    label_specs = (
        ("X (mm)", x_edge, X3D_LABEL_T),
        ("Y (mm)", y_edge, Y3D_LABEL_T),
        ("Z (mm)", z_edge, Z3D_LABEL_T),
    )

    for label, edge, t in label_specs:
        pos, angle = _axis_parallel_label_position(
            ax,
            edge[0],
            edge[1],
            t=t,
            offset=AXIS_PARALLEL_LABEL_OFFSET_3D,
        )
        ax.text2D(
            float(pos[0]),
            float(pos[1]),
            label,
            transform=ax.transAxes,
            rotation=angle,
            rotation_mode="anchor",
            ha="center",
            va="center",
            fontsize=FONT_SIZE_PT - 1,
            clip_on=False,
        )

def set_3d_limits_and_aspect(
    ax,
    point_cloud_xyz: np.ndarray,
    actual_trajectory: np.ndarray | None,
    planned_trajectory: np.ndarray,
    control_points: np.ndarray | None,
) -> None:
    reference_actual = actual_trajectory if actual_trajectory is not None else planned_trajectory
    reference_control = (
        np.atleast_2d(control_points)
        if control_points is not None and np.size(control_points) > 0
        else np.vstack([planned_trajectory[0], planned_trajectory[-1]])
    )

    planned_xyz = np.c_[np.zeros(len(planned_trajectory)), planned_trajectory]
    actual_xyz = np.c_[np.zeros(len(reference_actual)), reference_actual]
    control_xyz = np.c_[np.zeros(len(reference_control)), reference_control]
    all_xyz = np.vstack([point_cloud_xyz, planned_xyz, actual_xyz, control_xyz])

    xyz_min = np.nanmin(all_xyz, axis=0)
    xyz_max = np.nanmax(all_xyz, axis=0)
    xyz_range = xyz_max - xyz_min
    xyz_range[xyz_range <= 1e-9] = 1.0

    margin = LIMITS_3D_MARGIN_RATIO * xyz_range
    ax.set_xlim(xyz_min[0] - margin[0], xyz_max[0] + margin[0])
    ax.set_ylim(xyz_min[1] - margin[1], xyz_max[1] + margin[1])
    ax.set_zlim(xyz_min[2] - margin[2], xyz_max[2] + margin[2])

    try:
        if USE_DATA_ASPECT_3D:
            ax.set_box_aspect(tuple(xyz_range))
        else:
            ax.set_box_aspect(MANUAL_BOX_ASPECT_3D)
    except AttributeError:
        pass


def unique_legend_items(ax) -> tuple[list, list[str]]:
    """Return legend handles and labels while preserving first occurrence order."""
    handles, labels = ax.get_legend_handles_labels()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if not label or label.startswith("_") or label in unique_labels:
            continue
        unique_handles.append(handle)
        unique_labels.append(label)
    return unique_handles, unique_labels


def add_panel_legends(fig, ax2d, ax3d) -> None:
    """Add separate legends aligned by figure coordinates."""
    handles2d, labels2d = unique_legend_items(ax2d)
    if handles2d:
        fig.legend(
            handles2d,
            labels2d,
            loc="upper center",
            bbox_to_anchor=(LEGEND2D_FIG_X, LEGEND_FIG_Y),
            bbox_transform=fig.transFigure,
            ncol=LEGEND2D_NCOL,
            frameon=False,
            handlelength=1.7,
            handletextpad=0.45,
            columnspacing=0.95,
            labelspacing=0.35,
            borderaxespad=0.0,
        )

    handles3d, labels3d = unique_legend_items(ax3d)
    if handles3d:
        fig.legend(
            handles3d,
            labels3d,
            loc="upper center",
            bbox_to_anchor=(LEGEND3D_FIG_X, LEGEND_FIG_Y),
            bbox_transform=fig.transFigure,
            ncol=LEGEND3D_NCOL,
            frameon=False,
            handlelength=1.7,
            handletextpad=0.45,
            columnspacing=0.95,
            labelspacing=0.35,
            borderaxespad=0.0,
        )


def add_panel_labels(fig) -> None:
    """Add panel labels at exactly the same vertical level."""
    fig.text(
        # AX2D_POSITION[0] - PANEL_LABEL_X_OFFSET,
        AX2D_POSITION[0 ]- 0.08,
        PANEL_LABEL_FIG_Y,
        "B",
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_SIZE_PT,
        fontweight="bold",
    )
    fig.text(
        # AX3D_POSITION[0] - PANEL_LABEL_X_OFFSET,
        AX3D_POSITION[0] + 0.12,
        PANEL_LABEL_FIG_Y,
        "C",
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_SIZE_PT,
        fontweight="bold",
    )


def crop_point_cloud_for_plot(
    VIZ: ModuleType,
    point_cloud: np.ndarray,
    trajectory_reference: np.ndarray,
    trajectory_plan: np.ndarray | None = None,
    max_points: int = MAX_POINT_CLOUD_POINTS_3D,
    random_seed: int = POINT_CLOUD_RANDOM_SEED_3D,
) -> np.ndarray:
    """Crop the transformed point cloud near trajectories and optionally limit point count.

    Different from VIZ.crop_point_cloud_for_trajectory, this function does not
    down-sample by default. It also preserves RGB columns when the point cloud is
    Nx6 [X, Y, Z, R, G, B].

    Parameters
    ----------
    max_points:
        0 or a negative value keeps all cropped points. A positive value randomly
        keeps at most max_points rows.
    """
    points, colors = VIZ._split_xyz_rgb(point_cloud)

    traj = trajectory_reference if trajectory_plan is None else np.vstack([trajectory_reference, trajectory_plan])
    y_min, z_min = np.min(traj, axis=0)
    y_max, z_max = np.max(traj, axis=0)

    x_center = float(getattr(VIZ, "POINT_CLOUD_X_CENTER", 0.0))
    x_margin = float(getattr(VIZ, "POINT_CLOUD_X_MARGIN", 180.0))
    y_margin = float(getattr(VIZ, "POINT_CLOUD_Y_MARGIN", 320.0))
    z_margin = float(getattr(VIZ, "POINT_CLOUD_Z_MARGIN", 260.0))

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

    if max_points is not None and int(max_points) > 0 and cropped_points.shape[0] > int(max_points):
        rng = np.random.default_rng(random_seed)
        keep = np.sort(rng.choice(cropped_points.shape[0], size=int(max_points), replace=False))
        cropped_points = cropped_points[keep]
        cropped_colors = cropped_colors[keep] if cropped_colors is not None else None

    if cropped_colors is None:
        return cropped_points

    return np.c_[cropped_points, cropped_colors]



def plot_from_saved_sensor_csv(
    VIZ: ModuleType,
    planned_csv: Path,
    output_dir: Path,
    file_index: int,
    actual_sensor_csv: Path | None = None,
    point_cloud_file: Path | None = None,
    point_pose_file: Path | None = None,
    control_points_csv: Path | None = None,
    planned_skip_rows: int = 0,
    actual_skip_rows: int = 200,
    max_point_cloud_points: int = MAX_POINT_CLOUD_POINTS_3D,
    show: bool = True,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    planned_csv = Path(planned_csv)
    apply_nature_style()

    t_plan, dis_plan = read_sensor_like_csv(planned_csv, skip_rows=planned_skip_rows)
    planned_trajectory = VIZ.calc_true_trajectory(dis_plan)
    planned_duration = float(t_plan[-1] - t_plan[0])

    if point_cloud_file is None or point_pose_file is None:
        point_cloud_file, point_pose_file = point_cloud_files_for_index(file_index)
    point_cloud = VIZ.get_transformed_point_cloud(Path(point_cloud_file), Path(point_pose_file))

    actual_trajectory = None
    if actual_sensor_csv is not None:
        actual_trajectory = extract_actual_trajectory_for_duration(
            VIZ=VIZ,
            actual_sensor_csv=Path(actual_sensor_csv),
            point_cloud=point_cloud,
            planned_duration=planned_duration,
            actual_skip_rows=actual_skip_rows,
        )

    reference_for_surface = planned_trajectory if actual_trajectory is None else np.vstack([planned_trajectory, actual_trajectory])
    surface_fit = fit_surface_for_plot(VIZ, point_cloud, reference_for_surface)
    y_curve, z_curve = make_surface_curve(
        surface_fit,
        planned_trajectory,
        actual_trajectory if actual_trajectory is not None else planned_trajectory,
    )

    control_points = load_optional_control_points(control_points_csv, planned_csv)
    if control_points is not None and control_points.size:
        control_points = np.atleast_2d(control_points)

    crop_reference = actual_trajectory if actual_trajectory is not None else planned_trajectory
    cropped_points = crop_point_cloud_for_plot(
        VIZ=VIZ,
        point_cloud=point_cloud,
        trajectory_reference=crop_reference,
        trajectory_plan=planned_trajectory,
        max_points=max_point_cloud_points,
    )

    fig = plt.figure(figsize=(FIGURE_SIZE_CM[0] / 2.54, FIGURE_SIZE_CM[1] / 2.54), dpi=300)
    ax2d = fig.add_axes(AX2D_POSITION)
    ax3d = fig.add_axes(AX3D_POSITION, projection="3d")

    # Panel (a): 2D Y-Z view
    ax2d.plot(
        y_curve,
        z_curve,
        "-.",
        color=COLOR_SURFACE,
        linewidth=SURFACE_LINEWIDTH,
        label=LABEL_MATERIAL_SURFACE,
        zorder=1,
    )
    ax2d.plot(
        planned_trajectory[:, 0],
        planned_trajectory[:, 1],
        "-",
        color=COLOR_PLAN,
        linewidth=TRAJECTORY_LINEWIDTH_2D,
        label=LABEL_PLANNED_TRAJECTORY,
        zorder=3,
    )
    if actual_trajectory is not None:
        ax2d.plot(
            actual_trajectory[:, 0],
            actual_trajectory[:, 1],
            "-",
            color=COLOR_MEASURED,
            linewidth=TRAJECTORY_LINEWIDTH_2D,
            label=LABEL_MEASURED_TRAJECTORY,
            zorder=2,
        )

    if control_points is not None and control_points.size:
        ax2d.scatter(
            control_points[:, 0],
            control_points[:, 1],
            s=CONTROL_POINT_SIZE_2D,
            color=COLOR_CONTROL,
            edgecolors="white",
            linewidths=0.25,
            label=LABEL_CONTROL_POINTS,
            zorder=5,
        )

    ax2d.scatter(
        planned_trajectory[0, 0],
        planned_trajectory[0, 1],
        s=ENDPOINT_POINT_SIZE_2D,
        color=COLOR_START,
        edgecolors="white",
        linewidths=0.25,
        label=LABEL_START_POINT,
        zorder=6,
    )
    ax2d.scatter(
        planned_trajectory[-1, 0],
        planned_trajectory[-1, 1],
        s=ENDPOINT_POINT_SIZE_2D,
        color=COLOR_END,
        edgecolors="white",
        linewidths=0.25,
        label=LABEL_END_POINT,
        zorder=6,
    )
    ax2d.set_xlabel("Y (mm)")
    ax2d.set_ylabel("Z (mm)")
    ax2d.margins(x=0.05, y=0.08)
    style_2d_axes(ax2d)
    # Panel (b): 3D material face + trajectories
    if cropped_points.size:
        xyz, rgb = VIZ._split_xyz_rgb(cropped_points)
        scatter_kwargs = dict(
            s=POINT_CLOUD_POINT_SIZE_3D,
            alpha=POINT_CLOUD_ALPHA_3D,
            linewidths=0,
            label=LABEL_MATERIAL_FACE,
        )
        if USE_POINT_CLOUD_RGB_3D and rgb is not None:
            scatter_kwargs["c"] = rgb
        else:
            scatter_kwargs["c"] = COLOR_CLOUD
        ax3d.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], **scatter_kwargs)
    else:
        xyz = np.c_[np.zeros(len(planned_trajectory)), planned_trajectory]

    x_plan = np.zeros(planned_trajectory.shape[0])
    ax3d.plot(
        x_plan,
        planned_trajectory[:, 0],
        planned_trajectory[:, 1],
        "-",
        color=COLOR_PLAN,
        linewidth=TRAJECTORY_LINEWIDTH_3D,
        label=LABEL_PLANNED_TRAJECTORY,
        zorder=5,
    )

    if actual_trajectory is not None:
        x_actual = np.zeros(actual_trajectory.shape[0])
        ax3d.plot(
            x_actual,
            actual_trajectory[:, 0],
            actual_trajectory[:, 1],
            "-",
            color=COLOR_MEASURED,
            linewidth=TRAJECTORY_LINEWIDTH_3D,
            label=LABEL_MEASURED_TRAJECTORY,
            zorder=4,
        )

    if control_points is not None and control_points.size:
        x_ctrl = np.zeros(control_points.shape[0])
        ax3d.scatter(
            x_ctrl,
            control_points[:, 0],
            control_points[:, 1],
            s=CONTROL_POINT_SIZE_3D,
            color=COLOR_CONTROL,
            edgecolors="white",
            linewidths=0.20,
            label=LABEL_CONTROL_POINTS,
            zorder=6,
        )

    ax3d.scatter(
        [0.0],
        [planned_trajectory[0, 0]],
        [planned_trajectory[0, 1]],
        s=ENDPOINT_POINT_SIZE_3D,
        color=COLOR_START,
        edgecolors="white",
        linewidths=0.20,
        label=LABEL_START_POINT,
        zorder=7,
    )
    ax3d.scatter(
        [0.0],
        [planned_trajectory[-1, 0]],
        [planned_trajectory[-1, 1]],
        s=ENDPOINT_POINT_SIZE_3D,
        color=COLOR_END,
        edgecolors="white",
        linewidths=0.20,
        label=LABEL_END_POINT,
        zorder=7,
    )

    set_3d_limits_and_aspect(ax3d, xyz[:, :3], actual_trajectory, planned_trajectory, control_points)
    ax3d.view_init(
        elev=getattr(VIZ, "VIEW_3D_ELEV", 7),
        azim=getattr(VIZ, "VIEW_3D_AZIM", -40),
    )
    if USE_PROJECTED_AXIS_PARALLEL_LABELS_3D:
        ax3d.set_xlabel("")
        ax3d.set_ylabel("")
        ax3d.set_zlabel("")
    else:
        ax3d.set_xlabel("X (mm)")
        ax3d.set_ylabel("Y (mm)")
        ax3d.set_zlabel("Z (mm)")
    format_3d_axes(ax3d)
    add_projected_axis_parallel_3d_labels(ax3d)
    add_panel_labels(fig)
    add_panel_legends(fig, ax2d, ax3d)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_svg = output_dir / f"{planned_csv.stem}_redrawn_from_sensor_csv_nature_axis_parallel.svg"
    out_pdf = output_dir / f"{planned_csv.stem}_redrawn_from_sensor_csv_nature_axis_parallel.pdf"
    out_png = output_dir / f"{planned_csv.stem}_redrawn_from_sensor_csv_nature_axis_parallel.png"
    out_tiff = output_dir / f"{planned_csv.stem}_redrawn_from_sensor_csv_nature_axis_parallel.tiff"

    for out_file, fmt in ((out_svg, "svg"), (out_pdf, "pdf"), (out_png, "png"), (out_tiff, "tiff")):
        fig.savefig(out_file, format=fmt, dpi=DPI, bbox_inches=None, pad_inches=0.0, facecolor="white")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_svg, out_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a saved PGCD planned sensor-like CSV.")
    parser.add_argument("--file-index", type=int, default=DEFAULT_FILE_INDEX)
    parser.add_argument("--planned-csv", type=Path, default=None,
                        help="CSV saved by Plan_PGCD_current_material_save_sensor_csv.py.")
    parser.add_argument("--actual-sensor-csv", type=Path, default=None,
                        help="Optional measured sensor_data_*.csv. Default: experiment_data/CX_test/sensor_data_<file-index>.csv")
    parser.add_argument("--no-actual", action="store_true",
                        help="Only plot the planned CSV, without measured trajectory comparison.")
    parser.add_argument("--point-cloud-file", type=Path, default=None)
    parser.add_argument("--point-pose-file", type=Path, default=None)
    parser.add_argument("--control-points-csv", type=Path, default=None,
                        help="Optional Y,Z control-point CSV. Default: <planned-csv-stem>_control_points_yz.csv if it exists.")
    parser.add_argument("--viz-script", type=Path, default=None,
                        help="Optional path to Compare_trajectory_pointcloud_v9.py or v8.py.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--planned-skip-rows", type=int, default=0)
    parser.add_argument("--actual-skip-rows", type=int, default=200)
    parser.add_argument(
        "--max-point-cloud-points",
        type=int,
        default=MAX_POINT_CLOUD_POINTS_3D,
        help="Maximum cropped point-cloud points to draw. Use 0 to keep all cropped points without down-sampling.",
    )
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    VIZ = load_viz_module(args.viz_script)

    planned_csv = args.planned_csv
    if planned_csv is None:
        planned_csv = default_planned_csv(args.file_index, args.output_dir)

    actual_sensor_csv = None
    if not args.no_actual:
        actual_sensor_csv = args.actual_sensor_csv if args.actual_sensor_csv is not None else sensor_file_for_index(args.file_index)

    out_svg, out_png = plot_from_saved_sensor_csv(
        VIZ=VIZ,
        planned_csv=planned_csv,
        output_dir=args.output_dir,
        file_index=args.file_index,
        actual_sensor_csv=actual_sensor_csv,
        point_cloud_file=args.point_cloud_file,
        point_pose_file=args.point_pose_file,
        control_points_csv=args.control_points_csv,
        planned_skip_rows=args.planned_skip_rows,
        actual_skip_rows=args.actual_skip_rows,
        max_point_cloud_points=args.max_point_cloud_points,
        show=not args.no_show,
    )

    print(f"Read planned CSV: {planned_csv}")
    if actual_sensor_csv is not None:
        print(f"Read measured CSV: {actual_sensor_csv}")
    print(f"Point-cloud max points: {args.max_point_cloud_points} (0 means all cropped points)")
    print(f"Saved: {out_svg}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
