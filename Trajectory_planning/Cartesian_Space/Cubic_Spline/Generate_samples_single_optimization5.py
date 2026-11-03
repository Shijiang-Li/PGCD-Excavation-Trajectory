"""
鍗曟浼樺寲锛岀粰瀹氱墿鏂欒〃闈俊鎭笌鎸栨帢璧锋鐐癸紝鑾峰彇鏈€浼樿建杩广€備綔涓氶樆鍔涢娴嬫ā鍨嬬敤鐨勫浠诲姟妯″瀷锛堟繁搴﹀涔犳ā鍨嬶級銆?
"""
import random
import time

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
import random
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
import machine_params
from PhysicsModel.PointCloudFit import *
from Data_Preprocess_3_Class_P_Check_no_excavation_filter.Data_Preprocess_with_model_with_volume import \
    Get_material_face_fun, calc_pos_kxi, calc_v, calc_delta_Vg, get_point_cloud_fit
from PSO_multi import PSO
from Forward_model_diff.forward_model3_multi_task.model_multi_task_BiGRU_TCN import (BiGRU_Block, Chomp1d,
                                                                                     TemporalBlock, TemporalConvNet,
                                                                                     MLP_Block, BiGRU_TCN_Multi_model)
from Volume_prediction_model.Volume_model3_no_dis.model_InceptionTimeCBAM import (InceptionTimeChannelAttention,
                                                                                  InceptionTimeSpatialAttention,
                                                                                  InceptionTimeCBAM,
                                                                                  InceptionModuleCBAM,
                                                                                  InceptionTimeRegressionCBAM)


def plot_opt(loss):
    config = {
        "font.family": 'serif',
        "font.size": 8,
        "mathtext.fontset": 'stix',
        "font.serif": ['Times New Roman'],
    }
    rcParams.update(config)
    # 鍔犺浇Times New Roman瀛椾綋
    font_path = r'..\..\..\times.ttf'  # 灏嗘澶勭殑璺緞鏇挎崲涓篢imes New Roman瀛椾綋鏂囦欢鐨勫疄闄呰矾寰?
    font_prop = fm.FontProperties(fname=font_path, size=8)

    plt.figure(figsize=(7 / 2.54, 5 / 2.54))
    plt.plot(loss)  # pso.gbest_y_hist 淇濆瓨浜嗘瘡涓€浠ｇ殑鏈€浣虫崯澶卞€?
    # plt.title("Convergence Curve of PSO", fontproperties=font_prop)
    plt.xticks(np.arange(0, 31, 5))
    plt.yticks(np.arange(0.35, 0.471, 0.03))
    plt.xlabel("Iteration", fontproperties=font_prop)
    plt.ylabel("Fitness", fontproperties=font_prop)
    plt.grid()

    plt.tight_layout()
    plt.savefig('./Figure/pso_iteration.svg', format='svg')
    plt.show()


def get_mat_face_func1(pointCloud, offset, degree=4):
    return fit_material_surface(pointCloud, offset=offset, degree=degree)


def get_reference_trajectory(trajectory_distance):
    kine = Kinematic()
    length_data = len(trajectory_distance)
    # 閾叉枟鏂楀皷浣嶇疆
    position_bucket = np.zeros((length_data, 6))
    for i in range(length_data):
        position_bucket[i, :] = kine.kine_qdTowz(theta1=90, dbx=trajectory_distance[i, 0],
                                                 dgx=trajectory_distance[i, 1],
                                                 cdx=trajectory_distance[i, 2])
    return position_bucket


def plot_cartesian_figure(trajectory, pos, control_points, start_pos, end_pos, t, velocity, acceleration, para):
    """
    鐢诲嚭绗涘崱灏旂┖闂磋建杩广€侀€熷害涓庡姞閫熷害
    :param trajectory: 瑙勫垝鍑烘潵鐨勮建杩?
    :param pos: 杩愬姩瀛﹂€嗏€斺€?姝ｈВ姹傚緱鐨勮建杩?
    :param control_points: 涓棿鎺у埗鐐?
    :param start_pos: 璧峰鐐?
    :param end_pos: 缁堟鐐?
    :param t: 鏃堕棿
    :param velocity: 杞ㄨ抗閫熷害
    :param acceleration:  杞ㄨ抗鍔犻€熷害
    :param para:  鐗╂枡琛ㄩ潰鍙傛暟
    :return:
    """

    get_face_fun = Get_material_face_fun(para)

    mat_face_y = np.arange(end_pos[0] - 0.2 * (start_pos[0] - end_pos[0]),
                           start_pos[0] + 0.2 * (start_pos[0] - end_pos[0]))
    mat_face_z = np.zeros([mat_face_y.shape[0]])
    for i in range(mat_face_y.shape[0]):
        mat_face_z[i] = get_face_fun.matFace(mat_face_y[i])

    # 鍙鍖栬建杩广€侀€熷害鍜屽姞閫熷害
    fig, axs = plt.subplots(3, 1, figsize=(6, 9))

    # 杞ㄨ抗鍥?
    axs[0].plot(mat_face_y, mat_face_z, '-.', label='Material Face', linewidth=1)
    axs[0].plot(trajectory[:, 0], trajectory[:, 1], '--', label='Cubic Spline Trajectory', linewidth=3)
    axs[0].plot(pos[:, 1], pos[:, 2], label='Trajectory')
    axs[0].scatter(control_points[:, 0], control_points[:, 1], color='red', label='Control Points')
    axs[0].scatter(start_pos[0], start_pos[1], color='green', label='Start Position')
    axs[0].scatter(end_pos[0], end_pos[1], color='blue', label='End Position')
    axs[0].set_xlabel('Y')
    axs[0].set_ylabel('Z (mm)')
    axs[0].set_title('Trajectory')
    axs[0].legend(frameon=False)
    axs[0].grid()

    # 閫熷害鍥?
    axs[1].plot(t, velocity[:, 0], label='Velocity in X')
    axs[1].plot(t, velocity[:, 1], label='Velocity in Y')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('V (mm/s)')
    axs[1].set_title('Velocity')
    axs[1].legend(frameon=False)
    axs[1].grid()

    # 鍔犻€熷害鍥?
    axs[2].plot(t, acceleration[:, 0], label='Acceleration in X')
    axs[2].plot(t, acceleration[:, 1], label='Acceleration in Y')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Acceleration (mm^2/s)')
    axs[2].set_title('Acceleration')
    axs[2].legend(frameon=False)
    axs[2].grid()

    plt.tight_layout()
    plt.show()


def plot_cartesian_figure_ture(trajectory, pos, control_points, start_pos, end_pos, t, velocity, acceleration, para,
                               position_bucket):
    """
    鐢诲嚭绗涘崱灏旂┖闂磋建杩?鍖呮嫭鐪熷疄杞ㄨ抗)銆侀€熷害涓庡姞閫熷害
    :param trajectory: 瑙勫垝鍑烘潵鐨勮建杩?
    :param pos: 杩愬姩瀛﹂€嗏€斺€?姝ｈВ姹傚緱鐨勮建杩?
    :param control_points: 涓棿鎺у埗鐐?
    :param start_pos: 璧峰鐐?
    :param end_pos: 缁堟鐐?
    :param t: 鏃堕棿
    :param velocity: 杞ㄨ抗閫熷害
    :param acceleration:  杞ㄨ抗鍔犻€熷害
    :param para:  鐗╂枡琛ㄩ潰鍙傛暟
    :param position_bucket:
    :return:
    """

    get_face_fun = Get_material_face_fun(para)

    mat_face_y = np.arange(end_pos[0] - 0.2 * (start_pos[0] - end_pos[0]),
                           start_pos[0] + 0.2 * (start_pos[0] - end_pos[0]))
    mat_face_z = np.zeros([mat_face_y.shape[0]])
    for i in range(mat_face_y.shape[0]):
        mat_face_z[i] = get_face_fun.matFace(mat_face_y[i])

    # 鍙鍖栬建杩广€侀€熷害鍜屽姞閫熷害
    fig, axs = plt.subplots(3, 1, figsize=(6, 9))

    # 杞ㄨ抗鍥?
    axs[0].plot(position_bucket[:, 1], position_bucket[:, 2], '-.', label='True Trajectory', linewidth=1)
    axs[0].plot(mat_face_y, mat_face_z, '-.', label='Material Face', linewidth=1)
    axs[0].plot(trajectory[:, 0], trajectory[:, 1], '--', label='Cubic Spline Trajectory', linewidth=3)
    axs[0].plot(pos[:, 1], pos[:, 2], label='Trajectory')
    axs[0].scatter(control_points[:, 0], control_points[:, 1], color='red', label='Control Points')
    axs[0].scatter(start_pos[0], start_pos[1], color='green', label='Start Position')
    axs[0].scatter(end_pos[0], end_pos[1], color='blue', label='End Position')
    axs[0].set_xlabel('Y')
    axs[0].set_ylabel('Z (mm)')
    axs[0].set_title('Trajectory')
    axs[0].legend(frameon=False)
    axs[0].grid()

    # 閫熷害鍥?
    axs[1].plot(t, velocity[:, 0], label='Velocity in X')
    axs[1].plot(t, velocity[:, 1], label='Velocity in Y')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('V (mm/s)')
    axs[1].set_title('Velocity')
    axs[1].legend(frameon=False)
    axs[1].grid()

    # 鍔犻€熷害鍥?
    axs[2].plot(t, acceleration[:, 0], label='Acceleration in X')
    axs[2].plot(t, acceleration[:, 1], label='Acceleration in Y')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Acceleration (mm^2/s)')
    axs[2].set_title('Acceleration')
    axs[2].legend(frameon=False)
    axs[2].grid()

    plt.tight_layout()
    plt.show()


def plot_qd_figure(t, dis):
    """
    # 鍙鍖栬建杩广€侀€熷害鍜屽姞閫熷害
    :param t: 鏃堕棿
    :param dis: 椹卞姩娌圭几浣嶇Щ
    :return:
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 6))

    for i in range(3):
        if i == 0:
            title = 'db'
        elif i == 1:
            title = 'dg'
        else:
            title = 'cd'
        axes[0, i].plot(t, dis[:, i])
        axes[0, i].set_xlabel('t')
        axes[0, i].set_ylabel('y (mm)')
        axes[0, i].set_title(title + ' d')

        spline = UnivariateSpline(t, dis[:, i], s=0)  # s=0琛ㄧず閫氳繃鎵€鏈夌偣
        v = spline.derivative(n=1)(t)  # 涓€闃跺鏁颁负閫熷害
        axes[1, i].plot(t, v)
        axes[1, i].set_xlabel('t')
        axes[1, i].set_ylabel('v (mm/s)')
        axes[1, i].set_title(title + ' v')

        a = spline.derivative(n=2)(t)  # 浜岄樁瀵兼暟涓哄姞閫熷害
        axes[2, i].plot(t, a)
        axes[2, i].set_xlabel('t')
        axes[2, i].set_ylabel('a (mm^2/s)')
        axes[2, i].set_title(title + ' a')

    plt.tight_layout()
    plt.show()


def trajectory_planning_cubic_spline(start_pos, middle_pos, end_pos, start_angle, end_angle, t_end):
    """
    鏍规嵁鎺у埗鐐广€佽捣姝㈢偣濮挎€佽銆佹椂闂磋繘琛岃建杩硅鍒?
    :param start_pos: 璧峰鐐逛綅缃紝褰㈢姸涓?2,)
    :param middle_pos: 涓棿鎺у埗鐐癸紝褰㈢姸涓哄舰鐘朵负(batch, 1, 3)
    :param end_pos: 缁堟鐐逛綅缃紝褰㈢姸涓?2, )
    :param start_angle: 璧峰鐐瑰Э鎬佽锛屽舰鐘朵负(batch, 1)
    :param end_angle: 缁堟鐐瑰Э鎬佽锛屽舰鐘朵负(batch, 1)
    :param t_end: 杞ㄨ抗鎸栨帢鎬绘椂闂达紝褰㈢姸涓?batch, 1)
    :return:
    """
    batch_size = middle_pos.shape[0]

    start_pos = np.repeat(start_pos.reshape(1, -1), batch_size, axis=0)
    end_pos = np.repeat(end_pos.reshape(1, -1), batch_size, axis=0)

    control_points = np.stack([start_pos, middle_pos[:, 0, :2], end_pos], axis=1)  # 褰㈢姸涓?batch, n_points, 2)

    t_control = np.stack([np.zeros(batch_size), middle_pos[:, 0, 2], t_end], axis=1)  # 褰㈢姸涓?batch, 3)

    # 鍦?start_angle 鍜?end_angle 涔嬮棿鐢熸垚涓€涓瓑闂撮殧鐨勮搴︽暟缁勶紝# 褰㈢姸涓?batch, 3)
    control_angles = np.linspace(start_angle, end_angle, num=control_points.shape[1]).T

    # 鎵归噺杞ㄨ抗瑙勫垝
    trajectories = []
    velocities = []
    accelerations = []
    angles = []
    angle_velocities = []
    t_s = []
    pos_s = []
    dis_s = []

    for i in range(batch_size):
        # 瀵规瘡涓牱鏈繘琛岃建杩硅鍒?
        cubic_spline_pos = CubicSpline(t_control[i], control_points[i], bc_type='clamped')
        cubic_spline_angle = CubicSpline(t_control[i], control_angles[i], bc_type='clamped')

        # 鏃堕棿鍙傛暟
        t = np.linspace(0, t_control[i, -1], num=100)

        # 璁＄畻杞ㄨ抗
        trajectory = cubic_spline_pos(t)  # 杞ㄨ抗鍑芥暟
        velocity = cubic_spline_pos(t, 1)  # 涓€闃跺鏁颁负閫熷害
        acceleration = cubic_spline_pos(t, 2)  # 浜岄樁瀵兼暟涓哄姞閫熷害

        tao = cubic_spline_angle(t)  # 濮挎€佽鍑芥暟
        tao_v = cubic_spline_angle(t, 1)  # 濮挎€佽涓€闃跺鏁颁负瑙掗€熷害

        kne = Kinematic()
        dis = kne.kine_wzToqd_batch(np.zeros(100), trajectory[:, 0], trajectory[:, 1], tao)  # 杩愬姩瀛﹂€嗚В锛屾眰椹卞姩娌圭几浣嶇Щ
        pos = kne.kine_qdTowz_batch(90, dis[:, 0], dis[:, 1], dis[:, 2])  # 杩愬姩瀛︽瑙ｏ紝鏍规嵁椹卞姩娌圭几浣嶇Щ姹傞摬鏂楁枟灏栦綅濮?

        trajectories.append(trajectory)
        velocities.append(velocity)
        accelerations.append(acceleration)
        angles.append(tao)
        angle_velocities.append(tao_v)
        pos_s.append(pos)
        dis_s.append(dis)
        t_s.append(t)

    # 灏嗙粨鏋滆浆鎹负numpy鏁扮粍
    trajectories = np.array(trajectories)  # 褰㈢姸涓?batch_size, 100, 2)
    velocities = np.array(velocities)  # 褰㈢姸涓?batch_size, 100, 2)
    accelerations = np.array(accelerations)  # 褰㈢姸涓?batch_size, 100, 2)
    angles = np.array(angles)  # 褰㈢姸涓?batch_size, 100)
    angle_velocities = np.array(angle_velocities)  # 褰㈢姸涓?batch_size, 100)
    pos_s = np.array(pos_s)  # 褰㈢姸涓?batch_size, 100, 6)
    dis_s = np.array(dis_s)  # 褰㈢姸涓?batch_size, 100, 3)
    t_s = np.array(t_s)  # 褰㈢姸涓?batch_size, 100)

    return pos_s, trajectories, velocities, accelerations, angles, angle_velocities, t_s, control_points, dis_s


def get_input_data_batch(para, t, tra, tao, material, dis, v_xy):
    """
    璁＄畻鎸栨帢娣卞害銆佽〃闈㈣搴﹀拰鍥存垚闈㈢Н

    :param para: 鐗╂枡琛ㄩ潰褰㈣矊鍙傛暟锛岀Щ褰㈢姸涓?batch, 5)
    :param t: 鏃堕棿锛岀Щ褰㈢姸涓?batch, 100)
    :param tra: 杞ㄨ抗锛岀Щ褰㈢姸涓?batch, 100, 2)
    :param tao: 濮挎€佽锛屽舰鐘朵负(batch, 100)
    :param material: 鐗╂枡绫诲埆鎸栨帢闅惧害锛屽舰鐘朵负(batch, 1)
    :param dis: 椹卞姩娌圭几浣嶏紝绉诲舰鐘朵负(batch, 100, 3)
    :param v_xy: xy鏂瑰悜閫熷害锛岀Щ褰㈢姸涓?batch, 100, 2)
    """
    batch_size, n_points = tra.shape[0], tra.shape[1]
    x = tra[..., 0]  # shape: (batch, n_points)

    para = np.repeat(para.reshape(1, -1), batch_size, axis=0)

    # # 璁＄畻琛ㄩ潰楂樺害 y = a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0
    powers = np.stack([x ** i for i in range(4, -1, -1)], axis=-1)  # shape: (batch, n_points, 5)
    surface_y = np.sum(powers * para[..., np.newaxis].transpose(0, 2, 1), axis=-1)  # shape: (batch, n_points)

    # 璁＄畻娣卞害 (琛ㄩ潰楂樺害 - 杞ㄨ抗楂樺害)
    depth = (surface_y - tra[..., 1]) * 0.001  # shape: (batch, n_points)

    # 璁＄畻琛ㄩ潰鏂滅巼 dy/dx = 4a4*x^3 + 3a3*x^2 + 2a2*x + a1
    derivative_coeffs = para[:, :-1] * np.array([4, 3, 2, 1])[None, :]  # shape: (batch, 4)
    derivative_powers = np.stack([x ** i for i in range(3, -1, -1)], axis=-1)  # shape: (batch, n_points, 4)
    slopes = np.sum(derivative_coeffs[:, None, :] * derivative_powers, axis=-1)  # shape: (batch, n_points)

    # 璁＄畻琛ㄩ潰瑙掑害
    alpha = atand(slopes)  # shape: (n_points,)

    # 璁＄畻鎸栨帢瑙掑害
    beta = 180 + tao - alpha

    # 璁＄畻姣忓鐩搁偦鐐逛箣闂寸殑闈㈢Н
    dx = x[:, :-1] - x[:, 1:]   # shape: (batch, n_points-1)
    avg_depth = (depth[:, 1:] + depth[:, :-1]) / 2  # shape: (batch, n_points-1)
    segment_areas = dx * avg_depth  # shape: (batch, n_points-1)

    # 璁＄畻绱Н闈㈢Н
    cumulative_areas = np.zeros((batch_size, n_points))  # shape: (batch, n_points)
    cumulative_areas[:, 1:] = np.cumsum(segment_areas, axis=1)  # 浠庣浜屼釜鐐瑰紑濮嬬疮绉?
    Vg = cumulative_areas * 0.3 * 1e-3

    material = np.repeat(np.array([material]), n_points, axis=0)  # 褰㈢姸涓?n_points, )
    material_expand = np.repeat(np.array([material]), batch_size, axis=0)  # 褰㈢姸涓?batch, n_points)

    # 鏍规嵁x銆亂鏂瑰悜閫熷害璁＄畻鍜岄€熷害
    v = np.linalg.norm(v_xy, axis=-1) * 1e-3

    # print(t.shape, depth.shape, beta.shape, Vg.shape, alpha.shape, material_expand.shape, dis.shape,
    # tao.shape, v.shape)
    # plt.plot(t[0], tao[0])
    # plt.show()

    t = t[..., np.newaxis]
    depth = depth[..., np.newaxis]
    beta = beta[..., np.newaxis]
    Vg = Vg[..., np.newaxis]
    alpha = alpha[..., np.newaxis]
    material_expand = material_expand[..., np.newaxis]
    tao = tao[..., np.newaxis]
    v = v[..., np.newaxis]

    input_data = np.concatenate([t, depth, beta, Vg, alpha, material_expand, dis, tao, v], axis=-1)

    return input_data


def normalize_data_volume(input_data):
    # 归一化（范围为保密参数，见 machine_params）
    norm = machine_params.get("normalization")
    data_min = np.array(norm["volume_min"])
    data_max = np.array(norm["volume_max"])

    # 鎵╁睍缁村害浣縟ata_min鍜宒ata_max鐨勫舰鐘跺尮閰嶆暟鎹?
    # (features,) -> (1, 1, features)
    data_min = data_min[np.newaxis, np.newaxis, :]
    data_max = data_max[np.newaxis, np.newaxis, :]

    # 闃叉闄ら浂閿欒锛屽max鍜宮in鐩哥瓑鐨勭壒寰佽繘琛屽鐞?
    denominator = (data_max - data_min)
    denominator = np.where(denominator == 0, 1.0, denominator)

    # 褰掍竴鍖栬绠?
    normalized_data = (input_data - data_min) / denominator

    return normalized_data


def normalize_data_force(input_data):
    # 归一化（范围为保密参数，见 machine_params）
    norm = machine_params.get("normalization")
    data_min = np.array(norm["force_min"])
    data_max = np.array(norm["force_max"])

    # 鎵╁睍缁村害浣縟ata_min鍜宒ata_max鐨勫舰鐘跺尮閰嶆暟鎹?
    # (features,) -> (1, 1, features)
    data_min = data_min[np.newaxis, np.newaxis, :]
    data_max = data_max[np.newaxis, np.newaxis, :]

    # 闃叉闄ら浂閿欒锛屽max鍜宮in鐩哥瓑鐨勭壒寰佽繘琛屽鐞?
    denominator = (data_max - data_min)
    denominator = np.where(denominator == 0, 1.0, denominator)

    # 褰掍竴鍖栬绠?
    normalized_data = (input_data - data_min) / denominator

    return normalized_data


def get_f_and_v_input(para, t, tra, tao, material, dis, v_xy):
    """
    璁＄畻妯″瀷杈撳叆鏁版嵁锛屽寘鎷墿鏂欏弬鏁?
    :param para: 鐗╂枡琛ㄩ潰褰㈣矊鍙傛暟锛岀Щ褰㈢姸涓?batch, 5)
    :param t: 鏃堕棿锛岀Щ褰㈢姸涓?batch, 100)
    :param tra: 杞ㄨ抗锛岀Щ褰㈢姸涓?batch, 100, 2)
    :param tao: 濮挎€佽锛屽舰鐘朵负(batch, 100)
    :param material: 鐗╂枡绫诲埆鎸栨帢闅惧害锛屽舰鐘朵负(batch, 1)
    :param dis: 椹卞姩娌圭几浣嶏紝绉诲舰鐘朵负(batch, 100, 3)
    :param v_xy: xy鏂瑰悜閫熷害锛岀Щ褰㈢姸涓?batch, 100, 2)
    :return:
    """
    kp = 1
    if material == "sand":  # 娌欏瓙鐨勭墿鏂欏弬鏁?
        kp = 1  # 宸叉煡锛歱hi
    elif material == "small_stone":  # 缁嗙煶瀛愮殑鐗╂枡鍙傛暟
        kp = 2  # 宸叉煡
    elif material == "stone":  # 鐭冲瓙鐨勭墿鏂欏弬鏁?
        kp = 3  # 宸叉煡
    elif material == "soil":  # 鍦熷￥鐨勭墿鏂欏弬鏁?
        kp = 1  # 宸叉煡锛歱hi
    else:
        print("鐗╂枡绉嶇被杈撳叆閿欒")

    # t, depth, beta, Vg, alpha, material, dis, tao, v = calculate_surface_metrics(para, tra, tao)

    # 鑾峰彇杈撳叆鍙傛暟[t, d_, beta_, Vg_, alpha_, kp_, distance, xi, v__]
    # input_data = get_input_data(distance=dis, xi=tao, para=para, kp__=material)
    input_data = get_input_data_batch(para, t, tra, tao, kp, dis, v_xy)

    volume_input = input_data[:, :, [1, 2, 3, 4, 5, 9, 10]]
    volume_input = normalize_data_volume(volume_input)

    force_input = input_data[:, :, 1:]  # 鑾峰彇杈撳叆鍙傛暟[d_, beta_, Vg_, alpha_, kp_, distance, xi, v__]
    force_input = normalize_data_force(force_input)

    return force_input, volume_input


def calc_resistance(force_input_data, model_F):
    """
    鏍规嵁鏈哄櫒瀛︿範妯″瀷璁＄畻浣滀笟闃诲姏
    :param force_input_data: 妯″瀷鐨勮緭鍏ユ暟鎹紝shape=(batch, seq_length, 10)
    :param model_F: 浣滀笟闃诲姏棰勬祴妯″瀷
    :return:
    """
    # 璁＄畻浣滀笟闃诲姏
    # 杞崲涓?torch.Tensor锛屽苟纭繚鏁版嵁绫诲瀷涓?float32
    input_tensor = torch.from_numpy(force_input_data.astype(np.float32))

    lengths = torch.tensor([x.shape[0] for x in force_input_data])

    # 璋冪敤妯″瀷杩涜鎵瑰鐞嗛娴?
    output_F = model_F(input_tensor, lengths).detach().numpy()

    output_Ft = output_F[:, :, 0]
    output_Fn = output_F[:, :, 1]
    output_Mr = output_F[:, :, 2]
    # plt.plot(range(output_Ft[0].shape[0]), output_Ft[0])
    # plt.plot(range(output_Ft[0].shape[0]), output_Fn[0])
    # plt.plot(range(output_Ft[0].shape[0]), output_Mr[0])
    # plt.show()
    # quit()
    return output_Ft, output_Fn, output_Mr


def calc_volume(volume_input_data, model_v):
    """
    璁＄畻棰勬祴鎸栨帢鐗╂枡浣撶Н
    :param volume_input_data: 妯″瀷杈撳叆
    :param model_v: 鎸栨帢鐗╂枡浣撶Н棰勬祴妯″瀷
    :return: output_v 鐗╂枡浣撶Н
    """
    model_v.eval()

    input_tensor = torch.from_numpy(volume_input_data.astype(np.float32))

    # 璋冪敤妯″瀷杩涜鎵瑰鐞嗛娴?
    output_v = model_v(input_tensor).detach().numpy()

    return output_v


def calc_energy_consumption(input_data, velocity, tao, tao_v, model_F, t):
    """
    璁＄畻杞ㄨ抗涓婄殑浣滀笟闃诲姏銆佹渶澶у姛鐜囧拰鎬昏兘鑰楋紙鎵归噺璁＄畻鐗堟湰锛?
    :param input_data: 妯″瀷鐨勮緭鍏ユ暟鎹?shape=(batch, seq_length, input_dim)
    :param velocity: 绗涘崱灏旂┖闂寸殑杞ㄨ抗鐐逛笂鐨勯€熷害 shape=(batch, seq_length, 2)
    :param tao: 閾叉枟濮挎€佽 shape=(batch, seq_length)
    :param tao_v: 閾叉枟濮挎€佽閫熺巼 shape=(batch, seq_length)
    :param model_F: 浣滀笟闃诲姏棰勬祴妯″瀷
    :param t: 鏃堕棿搴忓垪 shape=(batch, seq_length)
    :return: energy_power_max: 鏈€澶у姛鐜?shape=(batch,)
             total_energy: 鎬昏兘鑰?shape=(batch,)
    """

    # 璁＄畻浣滀笟闃诲姏 shape=(batch, seq_length)
    output_Ft, output_Fn, output_Mr = calc_resistance(input_data, model_F)

    # 灏嗚搴﹁浆鎹负寮у害骞惰绠椾笁瑙掑嚱鏁板€?
    sin_tao = np.sin(np.deg2rad(-tao))  # (batch, seq_length)
    cos_tao = np.cos(np.deg2rad(-tao))  # (batch, seq_length)

    # 璁＄畻鍔熺巼鍚勫垎閲?
    # 鍒囧悜鍔涘姛鐜?
    Ft_power = -output_Ft * (velocity[..., 0] * sin_tao +
                             velocity[..., 1] * cos_tao) * 1e-3

    # 娉曞悜鍔涘姛鐜?
    Fn_power = output_Fn * (velocity[..., 0] * cos_tao -
                            velocity[..., 1] * sin_tao) * 1e-3

    # 鍔涚煩鍔熺巼
    Mr_power = -output_Mr * tao_v * np.pi / 180

    # 璁＄畻鎬诲姛鐜?
    energy_power = Ft_power + Fn_power + Mr_power  # (batch, seq_length)

    # 璁＄畻鏃堕棿闂撮殧 shape=(batch, seq_length-1)
    dt = t[:, 1:] - t[:, :-1]

    # 璁＄畻姣忎釜鏃堕棿闂撮殧鐨勫钩鍧囧姛鐜?shape=(batch, seq_length-1)
    avg_power = (energy_power[:, 1:] + energy_power[:, :-1]) / 2

    # 璁＄畻鎬昏兘鑰楋細浣跨敤姊舰娉曞垯杩涜鏁板€肩Н鍒?
    total_energy = np.sum(avg_power * dt, axis=1)  # shape=(batch,)

    return total_energy


def calc_energy_power_batch(input_data, velocity, tao, tao_v, model_F):
    """
    鏍规嵁浣滀笟闃诲姏棰勬祴妯″瀷璁＄畻杞ㄨ抗涓婄殑浣滀笟闃诲姏锛屽苟鏈€鍚庤绠楄建杩逛笂鐨勬渶澶у姛鐜囷紝杩欐槸涓轰簡璁＄畻鍔熺巼绾︽潫鐢?
    :param input_data: 妯″瀷鐨勮緭鍏ユ暟鎹?shape=(batch, seq_length, features)
    :param velocity: 绗涘崱灏旂┖闂寸殑杞ㄨ抗鐐逛笂鐨勯€熷害 shape=(batch, seq_length, 2)
    :param tao: 閾叉枟濮挎€佽 shape=(batch, seq_length)
    :param tao_v: 閾叉枟濮挎€佽閫熺巼 shape=(batch, seq_length)
    :param model_F: 浣滀笟闃诲姏棰勬祴妯″瀷
    :return: energy_power_max 姣忎釜batch涓渶澶у姛鐜?
    """
    # 璁＄畻浣滀笟闃诲姏 shape=(batch, seq_length)
    output_Ft, output_Fn, output_Mr = calc_resistance(input_data, model_F)

    # 灏嗚搴﹁浆鎹负寮у害骞惰绠椾笁瑙掑嚱鏁板€?
    sin_tao = np.sin(np.deg2rad(-tao))  # (batch, seq_length)
    cos_tao = np.cos(np.deg2rad(-tao))  # (batch, seq_length)

    # 璁＄畻鍔熺巼鍚勫垎閲?
    # 鍒囧悜鍔涘姛鐜?
    Ft_power = -output_Ft * (velocity[..., 0] * sin_tao +
                             velocity[..., 1] * cos_tao) * 1e-3

    # 娉曞悜鍔涘姛鐜?
    Fn_power = output_Fn * (velocity[..., 0] * cos_tao -
                            velocity[..., 1] * sin_tao) * 1e-3

    # 鍔涚煩鍔熺巼
    Mr_power = -output_Mr * tao_v * np.pi / 180

    # 璁＄畻鎬诲姛鐜?
    energy_power_delta = Ft_power + Fn_power + Mr_power  # (batch, seq_length)

    # 璁＄畻姣忎釜batch涓殑鏈€澶у姛鐜?
    energy_power_max = np.max(energy_power_delta, axis=1)  # (batch,)

    return energy_power_max


def calc_loss_from_trajectory(t, force_input_data, volume_input_data, v, tao, tao_v, model_F, model_v,
                              max_volume, max_energy_power, t_end, lambda_t, lambda_v, lambda_e):
    """
    璁＄畻鏃堕棿銆佺墿鏂欐寲鎺樹綋绉€佽兘鑰楃浉鍏崇殑鎸囨爣
    :param t: 鏃堕棿
    :param force_input_data: 闃诲姏妯″瀷鐨勮緭鍏?
    :param volume_input_data: 浣撶Н妯″瀷鐨勮緭鍏?
    :param v: 杞ㄨ抗閫熷害
    :param tao: 閾叉枟濮挎€佽
    :param tao_v: 瑙掗€熷害
    :param model_F: 浣滀笟闃诲姏棰勬祴妯″瀷
    :param model_v: 鐗╂枡鎸栨帢浣撶Н棰勬祴妯″瀷
    :param max_volume: 鏈€澶ф寲鎺樹綋绉?
    :param max_energy_power: 鏈€澶ц兘鑰?
    :param t_end: 杞ㄨ抗鑰楁椂
    :param lambda_t: 鏃堕棿鐩爣鏉冮噸绯绘暟
    :param lambda_v: 浣撶Н鐩爣鏉冮噸绯绘暟
    :param lambda_e: 鑳借€楃洰鏍囨潈閲嶇郴鏁?
    :return:
    """
    energy_consumption = calc_energy_consumption(force_input_data, v, tao, tao_v, model_F, t)  # 璁＄畻鑳借€?
    output_v = calc_volume(volume_input_data, model_v).reshape(-1)  # 璁＄畻浣撶Н

    min_energy_consumption = 0  # 鏈€灏忚兘鑰?
    min_volume = 0  # 鏈€灏忔寲鎺樹綋绉?
    max_t = 10  # 鏈€闀胯€楁椂
    min_t = 3  # 鏈€鐭€楁椂

    te = (t_end - min_t) / (max_t - min_t)  # 鏃堕棿鐩爣
    ve = (max_volume - output_v) / (max_volume - min_volume)  # 鐗╂枡鎸栨帢浣撶Н鐩爣
    we = (energy_consumption - min_energy_consumption) / (max_energy_power * 5 - min_energy_consumption)  # 鑳借€楃洰鏍?

    object_f = lambda_t * te + lambda_v * ve + lambda_e * we  # 鎬荤洰鏍?

    return object_f


def objective_function(x, model_F, model_v, material, para, max_energy_power, max_volume, start_pos, end_pos,
                       end_angle, point_1, point_2, point_3, point_4,
                       lambda_t=0.33, lambda_v=0.33, lambda_e=0.33):
    """
    鐩爣鍑芥暟
    :param x: 寰呬紭鍖栧弬鏁帮紝y鍧愭爣銆亃鍧愭爣銆佽捣鐐瑰Э鎬佽搴ngle锛屾€绘椂闂磘锛屽舰鐘朵负(batch, 4)
    :param model_F:
    :param model_v:
    :param material: 褰㈢姸涓?1, )
    :param para: 褰㈢姸涓?5, )
    :param max_energy_power:
    :param max_volume:
    :param start_pos: 璧峰鐐瑰潗鏍囷紝 褰㈢姸涓?2, )
    :param end_pos: 缁堟鐐瑰潗鏍囷紝褰㈢姸涓?2, )
    :param end_angle: 缁堟鐐瑰Э鎬佽锛?褰㈢姸涓?1, )
    :param point_1: 褰㈢姸涓?2, )锛岀敱start_pos銆乪nd_pos璁＄畻寰楀埌
    :param point_2: 褰㈢姸涓?2, )
    :param point_3: 褰㈢姸涓?2, )
    :param point_4: 褰㈢姸涓?2, )
    :param lambda_t: 鏃堕棿鐩爣鏉冮噸绯绘暟
    :param lambda_v: 浣撶Н鐩爣鏉冮噸绯绘暟
    :param lambda_e: 鑳借€楃洰鏍囨潈閲嶇郴鏁?
    :return:
    """

    # 涓棿鎺у埗鐐?
    # n_points = (len(x) - 2) // 2
    # middle_pos = np.zeros((n_points, 3))
    # for j in range(n_points):
    #     middle_pos[j, 0] = x[j * 2]
    #     middle_pos[j, 1] = x[j * 2 + 1]
    #     middle_pos[j, 2] = (x[-1] / (n_points + 1)) * (j + 1)
    # t_end = x[-1]  # 杞ㄨ抗鏃堕棿
    # start_angle = x[-2]
    n_points = 1
    batch_size = x.shape[0]
    middle_pos = np.zeros((batch_size, n_points, 3))
    middle_pos[:, 0, 0] = x[:, 0]
    middle_pos[:, 0, 1] = x[:, 1]
    middle_pos[:, 0, 2] = x[:, -1] / 2

    t_end = x[:, -1]  # 杞ㄨ抗鏃堕棿
    start_angle = x[:, -2]

    # 杞ㄨ抗瑙勫垝
    pos, trajectory, velocity, acceleration, tao, tao_v, t, control_points, dis = \
        trajectory_planning_cubic_spline(start_pos, middle_pos, end_pos, start_angle, end_angle, t_end)

    # plot_cartesian_figure(trajectory, pos, control_points, start_pos, end_pos, t, velocity, acceleration, para)
    # plot_qd_figure(t, dis)

    force_input, volume_input = get_f_and_v_input(para, t, trajectory, tao, material, dis, velocity)

    all_constraints = constraints(x, velocity, tao, tao_v, t, dis, model_F, model_v, force_input, volume_input,
                                  para, max_energy_power, max_volume, start_pos, end_pos,
                                  point_1, point_2, point_3, point_4)

    # 妫€鏌ユ瘡涓牱鏈槸鍚︽弧瓒虫墍鏈夌害鏉?
    constraints_satisfied = np.all(all_constraints >= 0, axis=1)  # shape=(batch_size,)

    # 璁＄畻鐩爣鍑芥暟
    object_f = np.where(constraints_satisfied,
                        calc_loss_from_trajectory(t, force_input, volume_input, velocity, tao, tao_v, model_F, model_v,
                                                  max_volume, max_energy_power, t_end, lambda_t, lambda_v, lambda_e),
                        np.full(constraints_satisfied.shape, np.inf))

    return object_f


def constraint_kinematic_batch(t, dis):
    """
    杩愬姩瀛︾害鏉熺殑鎵归噺璁＄畻鐗堟湰

    鍙傛暟:
    t: 鏃堕棿锛宻hape=(batch, n_points)
    dis: 娌圭几浣嶇Щ锛宻hape=(batch, n_points, 3)

    杩斿洖:
    cons_kinematic: shape=(batch, 12) 姣忎釜batch鐨?2涓害鏉熸潯浠?
    """
    batch_size = dis.shape[0]

    # 瀹氫箟绾︽潫甯搁噺
    lim = machine_params.get("constraints")  # [动臂,斗杆,铲斗] 位移/速度/加速度限值（保密参数）
    max_dis = np.array(lim["max_dis"])
    min_dis = np.array(lim["min_dis"])
    max_v = np.array(lim["max_v"])
    max_a = np.array(lim["max_a"])

    # 鍒濆鍖栫害鏉熺粨鏋滄暟缁?(batch, 12)
    cons_kinematic = np.zeros((batch_size, 12))

    # 璁＄畻浣嶇Щ绾︽潫 (姣忎釜娌圭几鐨勬渶澶ф渶灏忎綅绉荤害鏉?鍔ㄨ噦銆佹枟鏉嗐€侀摬鏂楃殑鏈€澶т綅绉荤害鏉?
    cons_kinematic[:, 0] = max_dis[0] - np.max(dis[:, :, 0], axis=1)  # 鍔ㄨ噦鏈€澶?
    cons_kinematic[:, 1] = np.min(dis[:, :, 0], axis=1) - min_dis[0]  # 鍔ㄨ噦鏈€灏?
    cons_kinematic[:, 2] = max_dis[1] - np.max(dis[:, :, 1], axis=1)  # 鏂楁潌鏈€澶?
    cons_kinematic[:, 3] = np.min(dis[:, :, 1], axis=1) - min_dis[1]  # 鏂楁潌鏈€灏?
    cons_kinematic[:, 4] = max_dis[2] - np.max(dis[:, :, 2], axis=1)  # 閾叉枟鏈€澶?
    cons_kinematic[:, 5] = np.min(dis[:, :, 2], axis=1) - min_dis[2]  # 閾叉枟鏈€灏?

    # 璁＄畻閫熷害鍜屽姞閫熷害绾︽潫
    for b in range(batch_size):
        # 涓烘瘡涓壒娆″垱寤烘牱鏉″嚱鏁?
        splines = [
            UnivariateSpline(t[b], dis[b, :, i], s=0)
            for i in range(3)  # 3涓补缂?
        ]

        # 璁＄畻閫熷害绾︽潫 (绗?-8涓害鏉?
        for i, spline in enumerate(splines):
            v = spline.derivative(n=1)(t[b])  # 璁＄畻閫熷害
            cons_kinematic[b, 6 + i] = max_v[i] - np.max(np.abs(v))

        # 璁＄畻鍔犻€熷害绾︽潫 (绗?-11涓害鏉?
        for i, spline in enumerate(splines):
            a = spline.derivative(n=2)(t[b])  # 璁＄畻鍔犻€熷害
            cons_kinematic[b, 9 + i] = max_a[i] - np.max(np.abs(a))

    return cons_kinematic


def constraint_power(force_input, velocity, tao, tao_v, model_F, max_energy_power):
    """
    鍔熺巼绾︽潫
    :param force_input: 闃诲姏妯″瀷杈撳叆
    :param velocity: 杞ㄨ抗鐐逛竴闃跺鏁颁负閫熷害
    :param tao: 濮挎€佽
    :param tao_v: 濮挎€佽涓€闃跺鏁颁负瑙掗€熷害
    :param model_F: 闃诲姏妯″瀷
    :param max_energy_power: 鏈€澶у姛鐜?
    :return:
    """
    energy_power = calc_energy_power_batch(force_input, velocity, tao, tao_v, model_F)

    cons_power = np.array([max_energy_power - energy_power]).T
    return cons_power


def constraint_volume(volume_input, model_v, max_volume):
    """
    鐗╂枡鎸栨帢浣撶Н绾︽潫
    :param volume_input: 浣撶Н妯″瀷杈撳叆
    :param model_v: 浣撶Н妯″瀷
    :param max_volume: 鏈€澶ф寲鎺樹綋绉?
    :return:
    """

    volume = calc_volume(volume_input, model_v)
    return max_volume - volume


def constraint_trajectory(x, para, start_pos, end_pos, point_1, point_2, point_3, point_4):
    """
    杞ㄨ抗鎺у埗鐐圭害鏉燂紝鎺у埗鐐圭殑x鏂瑰悜闇€瑕侀€掑噺
    :param x: 寰呬紭鍖栧弬鏁?
    :param para: 鐗╂枡琛ㄩ潰鎷熷悎鍑芥暟鐨勫弬鏁?
    :param start_pos: 璧峰鐐?
    :param end_pos: 缁堟鐐?
    :param point_1:
    :param point_2:
    :param point_3:
    :param point_4:
    :return:
    """

    # 涓棿鎺у埗鐐?
    n_points = 1
    batch_size = x.shape[0]
    middle_pos = np.zeros((batch_size, n_points, 3))
    middle_pos[:, 0, 0] = x[:, 0]
    middle_pos[:, 0, 1] = x[:, 1]
    middle_pos[:, 0, 2] = x[:, -1] / 2

    # 璧风偣鍒扮涓€涓帶鍒剁偣
    cons_trajectory = [start_pos[0] - middle_pos[:, 0, 0]]

    # 鎺у埗鐐逛箣闂?
    for i in range(n_points - 1):
        cons_trajectory.append(middle_pos[:, i + 1, 0] - middle_pos[:, i, 0])

    # 鏈€鍚庝竴涓帶鍒剁偣鍒扮粓鐐?
    cons_trajectory.append(middle_pos[:, -1, 0] - end_pos[0])

    # 杈圭晫绾︽潫
    # 鑾峰彇鎵€鏈夋帶鍒剁偣鐨勮竟鐣屽€?
    y_boundary = get_boundary(
        middle_pos[:, :, 0].reshape(-1),  # 灞曞钩鎵€鏈墄鍧愭爣
        point_1.repeat(n_points, axis=0),  # 閲嶅杈圭晫鐐逛互鍖归厤鎺у埗鐐规暟閲?
        point_2.repeat(n_points, axis=0),
        point_3.repeat(n_points, axis=0),
        point_4.repeat(n_points, axis=0)
    ).reshape(batch_size, n_points)  # 閲嶅鍥?(batch, n_points)

    # 娣诲姞杈圭晫绾︽潫
    for i in range(n_points):
        cons_trajectory.append(middle_pos[:, i, 1] - y_boundary[:, i])

    # 杞ㄨ抗鎺у埗鐐硅鍦ㄧ墿鏂欒〃闈箣鍐?

    # 璁＄畻琛ㄩ潰楂樺害 y = a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0
    for i in range(n_points):
        x_m = middle_pos[:, i, 0]
        surface_y = (para[0] * x_m ** 4 +
                     para[1] * x_m ** 3 +
                     para[2] * x_m ** 2 +
                     para[3] * x_m +
                     para[4])
        cons_trajectory.append(surface_y - middle_pos[:, i, 1])

    return np.array(cons_trajectory).T


def constraints(x, velocity, tao, tao_v, t, dis, model_F, model_v, force_input, volume_input, para,
                max_energy_power, max_volume, start_pos, end_pos,
                point_1, point_2, point_3, point_4):
    """
    鎬荤害鏉熸潯浠?

    :param x: 寰呬紭鍖栧弬鏁?
    :param velocity:
    :param tao:
    :param tao_v:
    :param t:
    :param dis:
    :param model_F:
    :param model_v:
    :param force_input:
    :param volume_input:
    :param para: 鐗╂枡琛ㄩ潰鍑芥暟鍙傛暟
    :param max_energy_power: 鏈€澶у姛鐜囩害鏉?
    :param max_volume: 鏈€澶ф寲鎺樹綋绉害鏉?
    :param start_pos:
    :param end_pos:
    :param point_1:
    :param point_2:
    :param point_3:
    :param point_4:
    :return:
    """

    cons_k = constraint_kinematic_batch(t, dis)  # 杩愬姩瀛︾害鏉?shape=(100, 12)
    cons_p = constraint_power(force_input, velocity, tao, tao_v, model_F, max_energy_power)  # 鍔熺巼绾︽潫 shape=(100, 1)
    cons_v = constraint_volume(volume_input, model_v, max_volume)  # 鎸栨帢浣撶Н绾︽潫 shape=(100, 1)
    cons_t = constraint_trajectory(x, para, start_pos, end_pos, point_1, point_2, point_3,
                                   point_4)  # 杞ㄨ抗杈圭晫绾︽潫 shape=(100, 4)
    # all_constraints = [item for sublist in [cons_k, cons_p, cons_v, cons_t] for item in sublist]
    # print(cons_k.shape, cons_p.shape, cons_v.shape, cons_t.shape)

    # 灏嗘墍鏈夌害鏉熷爢鍙犳垚涓€涓煩闃?
    constraints_matrix = np.concatenate([cons_k, cons_p, cons_v, cons_t], axis=1)  # shape=(batch_size, n_constraints)

    return constraints_matrix


def get_boundary(point_x, point_1, point_2, point_3, point_4):
    """
    纭繚鎺у埗鐐瑰湪鎸栨帢浠诲姟杈圭晫鍐咃紙鎵归噺璁＄畻鐗堟湰锛?
    :param point_x: 鎺у埗鐐?shape=(batch,)
    :param point_1: 杈圭晫鐐癸紝shape=(batch, 2)
    :param point_2: 杈圭晫鐐癸紝shape=(batch, 2)
    :param point_3: 杈圭晫鐐癸紝shape=(batch, 2)
    :param point_4: 杈圭晫鐐癸紝shape=(batch, 2)
    :return: y_boundary: shape=(batch,)
    """

    # 鍒濆鍖栫粨鏋滄暟缁?
    batch_size = point_x.shape[0]
    y_boundary = np.zeros(batch_size)

    # 鍒涘缓鍖洪棿鍒ゆ柇鐨勬帺鐮?
    mask_1_2 = (point_2[0] <= point_x) & (point_x <= point_1[0])
    mask_2_3 = (point_3[0] < point_x) & (point_x < point_2[0])
    mask_3_4 = (point_4[0] <= point_x) & (point_x <= point_3[0])

    def safe_divide(dy, dx):
        """瀹夊叏闄ゆ硶锛屽鐞嗛櫎鏁颁负闆剁殑鎯呭喌"""
        # 鍒涘缓鎺╃爜鏍囪瘑闄ゆ暟涓嶄负闆剁殑浣嶇疆
        nonzero_mask = np.abs(dx) > 1e-10
        # 鍒濆鍖栫粨鏋滄暟缁勪负闆?
        result = np.zeros_like(dx, dtype=np.float64)
        # 鍙湪闄ゆ暟涓嶄负闆剁殑浣嶇疆杩涜闄ゆ硶
        result[nonzero_mask] = dy[nonzero_mask] / dx[nonzero_mask]
        return result

    # 澶勭悊绗竴娈电嚎娈?
    dx_1_2 = point_2[0] - point_1[0]
    dy_1_2 = point_2[1] - point_1[1]
    slope_1_2 = safe_divide(dy_1_2, dx_1_2)
    y_1_2 = np.where(np.abs(dx_1_2) > 1e-10,
                     slope_1_2 * (point_x - point_1[0]) + point_1[1],
                     point_1[1])
    y_boundary = np.where(mask_1_2, y_1_2, y_boundary)

    # 澶勭悊绗簩娈电嚎娈?
    dx_2_3 = point_3[0] - point_2[0]
    dy_2_3 = point_3[1] - point_2[1]
    slope_2_3 = safe_divide(dy_2_3, dx_2_3)
    y_2_3 = np.where(np.abs(dx_2_3) > 1e-10,
                     slope_2_3 * (point_x - point_2[0]) + point_2[1],
                     point_2[1])
    y_boundary = np.where(mask_2_3, y_2_3, y_boundary)

    # 澶勭悊绗笁娈电嚎娈?
    dx_3_4 = point_4[0] - point_3[0]
    dy_3_4 = point_4[1] - point_3[1]
    slope_3_4 = safe_divide(dy_3_4, dx_3_4)
    y_3_4 = np.where(np.abs(dx_3_4) > 1e-10,
                     slope_3_4 * (point_x - point_3[0]) + point_3[1],
                     point_3[1])
    y_boundary = np.where(mask_3_4, y_3_4, y_boundary)

    return y_boundary


def get_start_end_pos(start_y, end_y, para):
    """
    鐢变簬鍙兘缁欏畾璧锋鐐箉鍧愭爣鐨勫€硷紝z鍧愭爣鍊奸渶瑕佹牴鎹墿鏂欏弬鏁拌繘琛岃绠?
    :param start_y:
    :param end_y:
    :param para: 鎷熷悎鐨勭墿鏂欒〃鏄庡嚱鏁板弬鏁?
    :return:
    """
    get_face_fun = Get_material_face_fun(para)

    start_z = get_face_fun.matFace(start_y)
    end_z = get_face_fun.matFace(end_y)

    start_pos = np.array([start_y, start_z])
    end_pos = np.array([end_y, end_z])
    return start_pos, end_pos


def define_control_para(start_y, end_y, end_angle, x, para):
    # start_y = 2450
    # end_y = 1750
    start_pos, end_pos = get_start_end_pos(start_y, end_y, para)

    # end_angle = -150

    # x = np.array([0.5 * (start_pos[0] + end_pos[0]) - 100,
    #               0.5 * (start_pos[1] + end_pos[1]) - 200, 1, 1, -50])

    # 涓棿鎺у埗鐐?
    n_points = len(x) // 3
    middle_pos = np.zeros((n_points, 3))
    for j in range(n_points):
        middle_pos[j, 0] = x[j * 3]
        middle_pos[j, 1] = x[j * 3 + 1]
        middle_pos[j, 2] = x[j * 3 + 2]
    t_end = np.sum(middle_pos, axis=0)[2] + x[-2]  # 杞ㄨ抗鏃堕棿
    start_angle = x[-1]

    return start_pos, middle_pos, end_pos, start_angle, end_angle, t_end


def visualize_optimization_trajectory(best_position, start_y, end_y, end_angle, para, volume_input_true_dis):
    """
    鍙鍖栦紭鍖栧悗鐨勮建杩?
    :param best_position:
    :param start_y:
    :param end_y:
    :param end_angle:
    :param para:
    :param volume_input_true_dis:
    :return:
    """
    x = best_position

    position_bucket = get_reference_trajectory(volume_input_true_dis)

    start_pos, middle_pos, end_pos, start_angle, end_angle, t_end = define_control_para(start_y, end_y, end_angle, x,
                                                                                        para)
    pos, trajectory, velocity, acceleration, tao_planned, tao_v, t, control_points, dis = \
        trajectory_planning_cubic_spline(start_pos, middle_pos, end_pos, start_angle, end_angle, t_end)

    plot_cartesian_figure_ture(trajectory, pos, control_points, start_pos, end_pos, t, velocity, acceleration, para,
                               position_bucket)


def single_optimization(file_index, random_start_y, random_end_y, end_angle, material, offset,
                        original_point_cloud_file_path_, original_point_position_file_, volume_input_true_dis_file):
    """
    缁欏畾璧风偣銆佺粓鐐广€佺粓鐐圭殑鎸栨帢瑙掑害
    寰呬紭鍖栫洰鏍囷細n涓帶鍒剁偣鐨?y z t锛屽墠涓€涓帶鍒剁偣鍒扮粓姝㈢偣鐨勬椂闂达紝璧峰鐐规寲鎺樿搴?
    :param file_index: 鏂囦欢搴忓彿
    :param random_start_y: 鍦ㄨ捣鐐瑰姞涓婁竴涓殢鏈鸿窛绂伙紝鐢ㄤ簬涓板瘜鏍锋湰
    :param random_end_y: 鍦ㄧ粓鐐瑰姞涓婁竴涓殢鏈鸿窛绂伙紝鐢ㄤ簬涓板瘜鏍锋湰
    :param end_angle: 闅忔満缁堢偣鎸栨帢瑙掑害锛岀敤浜庝赴瀵屾牱鏈?
    :param material: 鏉愭枡绫诲瀷锛屾敼鍙樻潗鏂欑被鍨嬫椂锛岃繕闇€瑕佹敼鍙樻鍑芥暟涓嬬殑鏂囦欢璺緞鍒板搴旂殑鏉愭枡绫诲瀷
    :param offset: 鐗╂枡涓嚎鍋忕Щ浣嶇疆
    :param original_point_cloud_file_path_: 瀵瑰簲鏉愭枡涓嬬殑鏂囦欢璺緞
    :param original_point_position_file_: 瀵瑰簲鏉愭枡涓嬬殑鏂囦欢璺緞
    :param volume_input_true_dis_file: 瀵瑰簲鏉愭枡涓嬬殑鏂囦欢璺緞
    :return:
    """

    """==========================================鍔犺浇妯″瀷==========================================="""
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model_F = torch.load(machine_params.model("model_F"), weights_only=False).to(device)
    model_v = torch.load(machine_params.model("model_v"), weights_only=False).to(device)

    """==========================================缁欏畾鐗╂枡淇℃伅涓庢寲鎺樿捣姝㈢偣==========================================="""
    # 鑾峰彇鐗╃悊琛ㄩ潰鎷熷悎鍑芥暟鐨勫弬鏁?
    points_Cloud = get_point_cloud_fit(original_point_cloud_file_path_ + "Point_cloud_{}.ply".format(file_index),
                                       original_point_position_file_ + 'Point_cloud_position_{}.txt'.format(
                                           file_index))  # 鑾峰彇鐐逛簯鏁版嵁骞惰繘琛屽潗鏍囩郴杞崲
    para = get_mat_face_func1(points_Cloud, offset, degree=4)  # 4娆″椤瑰紡鎷熷悎锛岃繑鍥炲€间负鍥涙澶氶」寮忕郴鏁?

    # 鑾峰彇鐪熷疄鎸栨帢杞ㄨ抗涓嬬殑鎸栨帢璧锋鐐?
    # volume_input_true_dis_file = ('./../../../OriginalData'
    #                               '/All_material_data_excavation_3Classes_P_Check_excavation/Preprocessed_data'
    #                               '/Filter_data/Sand_data_filter/sand_data_')
    volume_input_true_dis = np.loadtxt(open(volume_input_true_dis_file + "{}.csv".format(file_index), "rb"),
                                       delimiter=",", skiprows=1, usecols=[13, 14, 15])
    kne = Kinematic()
    true_start_y = kne.kine_qdTowz(90, volume_input_true_dis[0, 0], volume_input_true_dis[0, 1],
                                   volume_input_true_dis[0, 2])[1]
    true_end_y = kne.kine_qdTowz(90, volume_input_true_dis[-1, 0], volume_input_true_dis[-1, 1],
                                 volume_input_true_dis[-1, 2])[1]

    # 璧锋鐐?
    start_y = true_start_y + random_start_y
    end_y = true_end_y + random_end_y
    start_pos, end_pos = get_start_end_pos(start_y, end_y, para)

    # 杈圭晫鐐癸紝鏍规嵁璧锋鐐圭‘瀹氳竟鐣?
    point_1 = np.array([start_y + 100, start_pos[1]])
    point_2 = np.array([start_y + 100, 0.5 * (start_pos[1] + end_pos[1]) - 400])
    point_3 = np.array([end_y - 100, 0.5 * (start_pos[1] + end_pos[1]) - 400])
    point_4 = np.array([end_y - 100, end_pos[1]])

    """==========================================绮掑瓙缇や紭鍖?=========================================="""
    max_energy_power = 800  # 鏈€澶у姛鐜囩害鏉?
    max_volume = 25  # 鏈€澶ф寲鎺樹綋绉害鏉?

    def objective_function_with_params(*args):
        return objective_function(*args, model_F=model_F, model_v=model_v, material=material, para=para,
                                  max_energy_power=max_energy_power, max_volume=max_volume, start_pos=start_pos,
                                  end_pos=end_pos, end_angle=end_angle, point_1=point_1, point_2=point_2,
                                  point_3=point_3, point_4=point_4, lambda_t=0.33, lambda_v=0.33, lambda_e=0.33)

    # n涓帶鍒剁偣鐨?y z锛屾寲鎺樻€绘椂闂磘
    lower_bounds = np.array([0.5 * (start_pos[0] + end_pos[0]) - 100,
                             0.5 * (start_pos[1] + end_pos[1]) - 200, -90, 4])  # 涓嬬晫
    upper_bounds = np.array([0.5 * (start_pos[0] + end_pos[0]) + 100,
                             0.5 * (start_pos[1] + end_pos[1]) - 150, -50, 10])  # 涓婄晫

    # 瀹氫箟 PSO 鍙傛暟锛屼紭鍖栨渶灏忓€?
    pso = PSO(func=objective_function_with_params, dim=lower_bounds.shape[0], pop=50, max_iter=30,
              lb=lower_bounds, ub=upper_bounds)

    # 杩愯 PSO
    best_position, best_value, loss = pso.run()

    # if best_value < 1000:
    #     best_value = best_value[0]

    print("鏈€浣宠建杩?", np.array2string(best_position, separator=' ', formatter={'float_kind': lambda x: f'{x:.5f}'}))
    print(f"鏈€灏忔崯澶?{best_value}")

    plot_opt(loss)

    # 缁樺埗鎹熷け鍊奸殢杩唬娆℃暟鍙樺寲鐨勫浘
    # plt.ion()
    # plt.plot(pso.gbest_y_hist)  # pso.gbest_y_hist 淇濆瓨浜嗘瘡涓€浠ｇ殑鏈€浣虫崯澶卞€?
    # plt.title("Convergence Curve of PSO")
    # plt.xlabel("Iteration")
    # plt.ylabel("Best Loss Value")
    # plt.grid(True)
    # plt.show()
    # plt.pause(1)
    # plt.close()
    # plot_opt(loss)

    return best_position, best_value, start_y, end_y, end_angle, para, volume_input_true_dis


def main():
    np.random.seed(42)
    random.seed(42)
    material = 'sand'  # 鏉愭枡绫诲瀷锛屾敼鍙樻潗鏂欑被鍨嬫椂锛岃繕闇€瑕佹敼鍙樻鍑芥暟涓嬬殑鏂囦欢璺緞鍒板搴旂殑鏉愭枡绫诲瀷

    if material == 'sand':
        original_point_cloud_file_path = './../../../OriginalData/original_data/Sand/Sand_Point/'
        original_point_position_file = './../../../OriginalData/original_data/Sand/Sand_Point_Position/'
        volume_input_true_dis_file = ('./../../../OriginalData'
                                      '/All_material_data_excavation_3Classes_P_Check_excavation/Preprocessed_data'
                                      '/Filter_data/Sand_data_filter/sand_data_')
    elif material == "small_stone":  # 缁嗙煶瀛?
        original_point_cloud_file_path = './../../../OriginalData/original_data/Small_Stone/Small_Stone_Point/'
        original_point_position_file = './../../../OriginalData/original_data/Small_Stone/Small_Stone_Point_Position/'
        volume_input_true_dis_file = ('./../../../OriginalData/All_material_data_excavation_3Classes_P_Check_excavation'
                                      '/Preprocessed_data/Filter_data/Small_stone_data_filter/small_stone_data_')
    elif material == "stone":  # 鐭冲瓙
        original_point_cloud_file_path = './../../../OriginalData/original_data/Stone/Stone_Point/'
        original_point_position_file = './../../../OriginalData/original_data/Stone/Stone_Point_Position/'
        volume_input_true_dis_file = ('./../../../OriginalData/All_material_data_excavation_3Classes_P_Check_excavation'
                                      '/Preprocessed_data/Filter_data/Stone_data_filter/stone_data_')
    elif material == "soil":  # 鍦熷￥
        original_point_cloud_file_path = './../../../OriginalData/original_data/Soil/Soil_Point/'
        original_point_position_file = './../../../OriginalData/original_data/Soil/Soil_Point_Position/'
        volume_input_true_dis_file = ('./../../../OriginalData/All_material_data_excavation_3Classes_P_Check_excavation'
                                      '/Preprocessed_data/Filter_data/Soil_data_filter/soil_data_')
    else:
        original_point_cloud_file_path = ''
        original_point_position_file = ''
        volume_input_true_dis_file = ''
        print("鐗╂枡绉嶇被杈撳叆閿欒")
        quit()

    # best_res = np.zeros((1, 14))
    file_index = 1
    random_index = 0

    if random_index == 0:
        random_start_y = 0
        random_end_y = 0
        offset = 0
        end_angle = -160
    else:
        random_start_y = np.random.randint(-50, 50)
        random_end_y = np.random.randint(-50, 50)
        offset = np.random.randint(-200, 200)
        end_angle = np.random.randint(-180, -160)
    # best_position: n涓帶鍒剁偣鐨?y z t锛屽墠涓€涓帶鍒剁偣鍒扮粓姝㈢偣鐨勬椂闂达紝璧峰鐐规寲鎺樿搴?
    print(f"\n{file_index, random_index}===================================================")
    start_time = time.time()
    single_optimization(file_index, random_start_y, random_end_y, end_angle, material, offset,
                        original_point_cloud_file_path, original_point_position_file, volume_input_true_dis_file)
    end_time = time.time()
    print(end_time - start_time)


if __name__ == '__main__':
    main()

