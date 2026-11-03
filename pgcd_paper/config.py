"""
全局配置：路径、各物料场景、目标权重/约束、扩散超参。
所有相对路径均相对于 CUBIC（脚本运行时会 os.chdir 到该目录，与你的 Generate_samples_* 一致）。
"""
import os
import sys
from pathlib import Path

# Platform root: the Cubic_Spline objective code, the resistance/volume model
# modules and the trained model_*.pkl weights. They ship with this repository, so
# the default is the repository itself. Override with PGCD_PLATFORM_ROOT to point
# at a full platform that also holds the raw OriginalData point clouds.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import machine_params  # noqa: E402  confidential machine parameters, decrypted on first use
_DEFAULT_PLATFORM = PROJECT_ROOT
ROOT = os.environ.get("PGCD_PLATFORM_ROOT", str(_DEFAULT_PLATFORM))
CUBIC = os.path.join(ROOT, "Trajectory_planning", "Cartesian_Space", "Cubic_Spline")
PKG = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("PGCD_RESULTS_DIR", str(PROJECT_ROOT / "paper_reproduction_results"))
os.makedirs(RESULTS, exist_ok=True)

DEV = "cpu"

# ---- 相对 CUBIC 的数据根 ----
ORIG = "./../../../OriginalData/original_data"
EXC = ("./../../../OriginalData/All_material_data_excavation_3Classes_P_Check_excavation"
       "/Preprocessed_data/Filter_data")

# ---- 各物料场景文件（取自 Generate_samples_*.main() 的 if-else） ----
MATERIALS = {
    "sand":        dict(cloud="Sand/Sand_Point",               pos="Sand/Sand_Point_Position",
                        truedir="Sand_data_filter",            trueprefix="sand_data_"),
    "small_stone": dict(cloud="Small_Stone/Small_Stone_Point", pos="Small_Stone/Small_Stone_Point_Position",
                        truedir="Small_stone_data_filter",     trueprefix="small_stone_data_"),
    "stone":       dict(cloud="Stone/Stone_Point",             pos="Stone/Stone_Point_Position",
                        truedir="Stone_data_filter",           trueprefix="stone_data_"),
    "soil":        dict(cloud="Soil/Soil_Point",               pos="Soil/Soil_Point_Position",
                        truedir="Soil_data_filter",            trueprefix="soil_data_"),
}

# ---- 预计算场景几何（料面多项式 + 起止点），使目标函数无需原始点云 ----
SCENE_CACHE = os.environ.get("PGCD_SCENE_CACHE", str(PROJECT_ROOT / "data" / "scene_cache.npz"))
USE_RAW_SCENES = os.environ.get("PGCD_USE_RAW_SCENES", "0") == "1"

# ---- 模型权重：阻力/满斗率模型为保密参数，加密存于 machine_params.enc ----

# ---- 目标函数权重与约束（取自他 main 默认） ----
LAMBDA = (0.33, 0.34, 0.33)   # (time, volume, energy)
# 峰值功率与最大体积限值为保密参数：machine_params.get("constraints")
END_ANGLE = -160

# ---- 决策变量 ----
DIM = 4   # [middle_y, middle_z, start_angle, t_end]

# ---- 扩散 / 训练超参 ----
T_DIFF = 100
BETA_START, BETA_END = 1e-4, 0.02
DIFF_STEPS = 6000
DIFF_BS = 256
DIFF_LR = 8e-4
BC_TEMP = 0.2                 # filtered-BC 温度（×std(J)）：→0 只学最优样本、→∞ 退化为普通 BC（0.2=最优 best+多样性甜点）

# ---- 数据集 ----
N_SAMPLES = 20000
EVAL_CHUNK = 512

# ---- 采样/评估 ----
N_SAMPLE_EVAL = 2048
