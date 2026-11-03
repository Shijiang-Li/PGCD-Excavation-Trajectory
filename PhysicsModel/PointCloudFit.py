import numpy as np
from PhysicsModel.Kinematics import *
try:
    import open3d as o3d
except ModuleNotFoundError:
    o3d = None


def _read_ply_points(point_cloud_file):
    if o3d is not None:
        return np.asarray(o3d.io.read_point_cloud(str(point_cloud_file)).points)

    header_lines = 0
    vertex_count = None
    properties = []
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
    usecols = [properties.index(name) for name in ("x", "y", "z")]
    points = np.loadtxt(point_cloud_file, skiprows=header_lines, max_rows=vertex_count, usecols=usecols)
    return np.atleast_2d(points)


def _read_ply_colors(point_cloud_file, point_count):
    if o3d is not None:
        return np.asarray(o3d.io.read_point_cloud(str(point_cloud_file)).colors)

    header_lines = 0
    vertex_count = None
    properties = []
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

    color_names = ("red", "green", "blue")
    if vertex_count is None or not all(name in properties for name in color_names):
        return np.zeros((point_count, 3), dtype=float)
    usecols = [properties.index(name) for name in color_names]
    colors = np.loadtxt(point_cloud_file, skiprows=header_lines, max_rows=vertex_count, usecols=usecols)
    return np.atleast_2d(colors) / 255.0


def _finite_yz(points):
    y = np.asarray(points[1, :], dtype=float).reshape(-1)
    z = np.asarray(points[2, :], dtype=float).reshape(-1)
    ok = np.isfinite(y) & np.isfinite(z)
    return y[ok], z[ok]


def _mad(values):
    values = np.asarray(values, dtype=float)
    med = np.median(values)
    return np.median(np.abs(values - med))


def _robust_yz_filter(y, z, max_mad=5.0):
    if y.size < 8:
        return y, z
    keep = np.ones(y.shape, dtype=bool)
    for values in (y, z):
        med = np.median(values)
        scale = 1.4826 * _mad(values)
        if scale > 1e-9:
            keep &= np.abs(values - med) <= max_mad * scale
    return y[keep], z[keep]


def _surface_bins(y, z, n_bins=80, min_per_bin=4, surface_quantile=0.55):
    if y.size < max(min_per_bin * 2, 12):
        return y, z
    edges = np.linspace(y.min(), y.max(), n_bins + 1)
    by, bz = [], []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (y >= left) & (y < right)
        if right == edges[-1]:
            mask = (y >= left) & (y <= right)
        if int(mask.sum()) < min_per_bin:
            continue
        by.append(float(np.median(y[mask])))
        bz.append(float(np.quantile(z[mask], surface_quantile)))
    if len(by) < 8:
        return y, z
    return np.asarray(by), np.asarray(bz)


def _polyfit_safe(y, z, degree):
    if y.size <= degree:
        raise ValueError(f"not enough material surface points for degree {degree}: {y.size}")
    return np.polyfit(y, z, degree)


def _ransac_polyfit(y, z, degree, max_trials=240, residual_threshold=None, random_state=0):
    if y.size <= degree + 2:
        return _polyfit_safe(y, z, degree)
    rng = np.random.RandomState(random_state)
    if residual_threshold is None:
        spread = max(float(np.ptp(z)), 1.0)
        residual_threshold = max(8.0, 2.5 * 1.4826 * _mad(z), 0.015 * spread)

    best_coef, best_inliers, best_score = None, None, (-1, np.inf)
    sample_size = degree + 1
    for _ in range(max_trials):
        idx = rng.choice(y.size, sample_size, replace=False)
        if np.ptp(y[idx]) < 1e-6:
            continue
        try:
            coef = np.polyfit(y[idx], z[idx], degree)
        except (np.linalg.LinAlgError, ValueError):
            continue
        residual = np.abs(np.polyval(coef, y) - z)
        inliers = residual <= residual_threshold
        score = (int(inliers.sum()), float(np.median(residual[inliers])) if inliers.any() else np.inf)
        if score[0] > best_score[0] or (score[0] == best_score[0] and score[1] < best_score[1]):
            best_coef, best_inliers, best_score = coef, inliers, score

    if best_inliers is None or int(best_inliers.sum()) <= degree:
        return _polyfit_safe(y, z, degree)
    return np.polyfit(y[best_inliers], z[best_inliers], degree)


def fit_material_surface(pointCloud, offset=0.0, degree=4, slice_half_width=5.0,
                         min_slice_points=80, max_slice_half_width=80.0):
    """Robustly fit z=f(y) near an x offset and return np.polyfit coefficients."""
    pc = np.asarray(pointCloud, dtype=float)
    if pc.ndim != 2 or pc.shape[0] < 3:
        raise ValueError(f"pointCloud must have shape (>=3, n), got {pc.shape}")

    half_width = float(slice_half_width)
    selected = None
    while half_width <= max_slice_half_width:
        mask = (pc[0, :] > offset - half_width) & (pc[0, :] < offset + half_width)
        if int(mask.sum()) >= min_slice_points or half_width >= max_slice_half_width:
            selected = pc[:, np.ravel(mask.reshape(-1))]
            break
        half_width *= 1.8
    if selected is None or selected.shape[1] == 0:
        selected = pc

    y, z = _finite_yz(selected)
    y, z = _robust_yz_filter(y, z)
    by, bz = _surface_bins(y, z)
    try:
        return _ransac_polyfit(by, bz, degree)
    except ValueError:
        return _ransac_polyfit(y, z, degree)


def trans_points_cloud_no_color(pointCloudFile, db_x, dg_x, cd_x):
    points = _read_ply_points(pointCloudFile)
    points[:, 0] = points[:, 0] - 60

    # 定义条件：y方向上小于等于600，x方向上小于等于-200大于等于200
    # condition_y = points[:, 1] <= 600
    condition_x = np.logical_and(points[:, 0] >= -200, points[:, 0] <= 200)

    # 组合条件
    # final_condition = np.logical_and(condition_y, condition_x)
    final_condition = condition_x

    # 筛选满足条件的点和颜色
    filtered_points = points[final_condition]

    pCloud = np.c_[filtered_points, np.ones(filtered_points.shape[0])]  # 添加一行单位向量，组成扩展矩阵，以便坐标变换
    pCloud = pCloud.T

    # -----------------------------------将点云从相机坐标系转换到基坐标系--------------------------
    k = Kinematic()
    pCloud = k.plyTrans(pCloud, 90, db_x, dg_x, cd_x)
    return pCloud


def trans_points_cloud(pointCloudFile, db_x, dg_x, cd_x):
    points = _read_ply_points(pointCloudFile)
    points[:, 0] = points[:, 0] - 60
    colors = _read_ply_colors(pointCloudFile, points.shape[0])

    # 定义条件：y方向上小于等于600，x方向上小于等于-200大于等于200
    # condition_y = points[:, 1] <= 600
    condition_x = np.logical_and(points[:, 0] >= -200, points[:, 0] <= 200)

    # 组合条件
    # final_condition = np.logical_and(condition_y, condition_x)
    final_condition = condition_x

    # 筛选满足条件的点和颜色
    filtered_points = points[final_condition]
    filtered_colors = colors[final_condition] * 255

    pCloud = np.c_[filtered_points, np.ones(filtered_points.shape[0])]  # 添加一行单位向量，组成扩展矩阵，以便坐标变换
    pCloud = pCloud.T

    # -----------------------------------将点云从相机坐标系转换到基坐标系--------------------------
    k = Kinematic()
    pCloud = k.plyTrans(pCloud, 90, db_x, dg_x, cd_x)
    return pCloud, filtered_colors.T


def trans_points_cloud_with_no_cut(pointCloudFile, db_x, dg_x, cd_x):
    points = _read_ply_points(pointCloudFile)
    points[:, 0] = points[:, 0] - 60
    colors = _read_ply_colors(pointCloudFile, points.shape[0]) * 255

    pCloud = np.c_[points, np.ones(points.shape[0])]  # 添加一行单位向量，组成扩展矩阵，以便坐标变换
    pCloud = pCloud.T

    # -----------------------------------将点云从相机坐标系转换到基坐标系--------------------------
    k = Kinematic()
    pCloud = k.plyTrans(pCloud, 90, db_x, dg_x, cd_x)
    return pCloud, colors.T


def trans_points_cloud_base_to_bucket(pCloud, db_x, dg_x, cd_x):
    # -----------------------------------将点云从基坐标系转换到铲斗坐标系--------------------------
    k = Kinematic()
    pCloud = k.base_to_bucket(pCloud, 90, db_x, dg_x, cd_x)
    return pCloud


def get_mat_face_func(pointCloud, degree=4):
    return fit_material_surface(pointCloud, offset=0.0, degree=degree)


# point_cloud_file = r'.\..\OriginalData\PointCloud.ply'
# points_Cloud = trans_points_cloud(point_cloud_file, db_x=100, dg_x=110, cd_x=100)
# para = get_mat_face_func(points_Cloud, degree=4)
# print(para)
# print(isinstance(para, np.ndarray))
