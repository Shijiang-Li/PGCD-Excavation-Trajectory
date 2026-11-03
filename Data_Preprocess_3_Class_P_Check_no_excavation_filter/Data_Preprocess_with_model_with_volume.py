import machine_params
import time

from PhysicsModel.Dynamic import DynamicsOfExcavator as dyn
from PhysicsModel.PointCloudFit import *
from DataProcess.DataProcessTool import *
from DataProcess.Pin_sensor_correct import correct_Mx, correct_My, correct_O3x, correct_O3y
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

dyn = dyn()


def data_filter(originalFile):
    """
    读取数据并进行滤波
    :param originalFile: 原始数据文件
    :return:
    """
    t_ = np.loadtxt(open(originalFile, "rb"), delimiter=",", skiprows=2, usecols=[0])  # 时间
    distance_ = np.loadtxt(open(originalFile, "rb"), delimiter=",", skiprows=2, usecols=[1, 2, 3])  # 位移传感器数据，动臂、斗杆、铲斗

    hydraulic_ = np.loadtxt(open(originalFile, "rb"), delimiter=",", skiprows=2, usecols=[4, 5, 6, 7, 8, 9])  # 油压数据
    rollOutput_ = np.loadtxt(open(originalFile, "rb"), delimiter=",", skiprows=2,
                             usecols=[10, 11, 12, 14])  # 销轴传感器数据Mx、My、O3x、O3y
    # sigma = 30  # 窗口大小
    # order = 0  # 多项式阶数
    # data_db = distance_[:, 0]
    # data_dg = distance_[:, 1]
    # data_cd = distance_[:, 2]
    # smoothed_data_db = gaussian_filter1d(data_db, sigma=sigma, order=order)
    # smoothed_data_dg = gaussian_filter1d(data_dg, sigma=sigma, order=order)
    # smoothed_data_cd = gaussian_filter1d(data_cd, sigma=sigma, order=order)
    # distance_ = np.c_[smoothed_data_db, smoothed_data_dg, smoothed_data_cd]

    # data_p_db_d = hydraulic_[:, 0]
    # data_p_db_x = hydraulic_[:, 1]
    # data_p_dg_d = hydraulic_[:, 2]
    # data_p_dg_x = hydraulic_[:, 3]
    # data_p_cd_d = hydraulic_[:, 4]
    # data_p_cd_x = hydraulic_[:, 5]
    # smoothed_data_p_db_d = gaussian_filter1d(data_p_db_d, sigma=sigma, order=order)
    # smoothed_data_p_db_x = gaussian_filter1d(data_p_db_x, sigma=sigma, order=order)
    # smoothed_data_p_dg_d = gaussian_filter1d(data_p_dg_d, sigma=sigma, order=order)
    # smoothed_data_p_dg_x = gaussian_filter1d(data_p_dg_x, sigma=sigma, order=order)
    # smoothed_data_p_cd_d = gaussian_filter1d(data_p_cd_d, sigma=sigma, order=order)
    # smoothed_data_p_cd_x = gaussian_filter1d(data_p_cd_x, sigma=sigma, order=order)
    # hydraulic_ = np.c_[smoothed_data_p_db_d, smoothed_data_p_db_x, smoothed_data_p_dg_d, smoothed_data_p_dg_x,
    #                    smoothed_data_p_cd_d, smoothed_data_p_cd_x]

    # data_Mx = rollOutput_[:, 0]
    # data_My = rollOutput_[:, 1]
    # data_O3x = rollOutput_[:, 2]
    # data_O3y = rollOutput_[:, 3]
    # smoothed_data_Mx = gaussian_filter1d(data_Mx, sigma=sigma, order=order)
    # smoothed_data_My = gaussian_filter1d(data_My, sigma=sigma, order=order)
    # smoothed_data_O3x = gaussian_filter1d(data_O3x, sigma=sigma, order=order)
    # smoothed_data_O3y = gaussian_filter1d(data_O3y, sigma=sigma, order=order)
    # rollOutput_ = np.c_[smoothed_data_Mx, smoothed_data_My, smoothed_data_O3x, smoothed_data_O3y]

    return t_, distance_, hydraulic_, rollOutput_


def calculate_theta_a(distance_):
    """
    计算theta关节角、角速度、角加速度
    :param distance_: 滤波后的油缸位移数据
    :return:
    """
    # 预先分配储存空间
    length = len(distance_)
    t_interval = 0.01  # 采样时间间隔
    tLong = t_interval * length  # 总时长
    sigma = np.zeros((length, 7))
    theta = np.zeros((length, 3))
    theta_t = np.zeros((length, 3))
    theta_tt = np.zeros((length, 3))

    n = 0
    for _ in np.arange(0, tLong, t_interval):
        theta[n, :], sigma[n, :] = dyn.calc_theta_sigma(distance_[n, 0], distance_[n, 1], distance_[n, 2])
        if n == 0:
            # 计算后一时刻角度
            theta_after, _ = dyn.calc_theta_sigma(distance_[n + 1, 0], distance_[n + 1, 1], distance_[n + 1, 2])

            theta_t[n, :] = (theta_after - theta[n, :]) / t_interval  # 计算后一时刻角速度

            theta_tt[n, :] = np.array([0, 0, 0])  # 角加速度设为0

        elif n == 1:
            # 计算前一时刻角度
            theta_before, _ = dyn.calc_theta_sigma(distance_[n - 1, 0], distance_[n - 1, 1], distance_[n - 1, 2])
            # 计算后一时刻角度
            theta_after, _ = dyn.calc_theta_sigma(distance_[n + 1, 0], distance_[n + 1, 1], distance_[n + 1, 2])

            theta_t_before = (theta[n, :] - theta_before) / t_interval  # 计算前一时刻角速度
            theta_t_after = (theta_after - theta[n, :]) / t_interval  # 计算后一时刻角速度
            theta_t[n, :] = (theta_t_before + theta_t_after) / 2  # 计算前一时刻与后一时刻的平均角速度

            theta_tt[n, :] = np.array((theta_t_after - theta_t_before) / t_interval)  # 计算角加速度
        else:
            # 计算前二时刻角度
            theta_bef_bef, _ = dyn.calc_theta_sigma(distance_[n - 2, 0], distance_[n - 2, 1], distance_[n - 2, 2])
            # 计算前一时刻角度
            theta_before, _ = dyn.calc_theta_sigma(distance_[n - 1, 0], distance_[n - 1, 1], distance_[n - 1, 2])

            theta_t_bef_bef = (theta_before - theta_bef_bef) / t_interval  # 计算前二时刻角速度
            theta_t_before = (theta[n, :] - theta_before) / t_interval  # 计算前一时刻角速度
            theta_t[n, :] = (theta_t_before + theta_t_bef_bef) / 2  # 计算前二时刻与前一时刻的平均角速度

            # theta_t[n, :] = normalize_data(theta_t[n, :], up_limit=5, down_limit=-5)  # 限定角加速度输出值

            theta_tt[n, :] = np.array((theta_t_before - theta_t_bef_bef) / t_interval)  # 计算角加速度
            # theta_tt[n, :] = normalize_data(theta_tt[n, :], up_limit=50, down_limit=-50)  # 限定角加速度输出值
        n = n + 1

    theta_tt = thetaAccFilter(theta_tt, win_length=53, k=3)  # 再次对角加速度数据滤波
    return theta, theta_t, theta_tt


def data_preprocess_and_calc_resistance(originalFile):
    # 对原始数据进行去异常值、滤波操作
    t, distance, hydraulic, rollOutput = data_filter(originalFile)
    # 根据空载数据对销轴传感器数据校正
    rollOutput[:, 0] = correct_Mx(rollOutput[:, 0])
    rollOutput[:, 1] = correct_My(rollOutput[:, 1])
    rollOutput[:, 2] = correct_O3x(rollOutput[:, 2])
    rollOutput[:, 3] = correct_O3y(rollOutput[:, 3])

    theta, theta_t, theta_tt = calculate_theta_a(distance)

    length = len(distance)
    accelerate = np.zeros((length, 9))
    FLoadCalc = np.zeros((length, 3))

    # 计算作业阻力，输入参数向量，rollOutput_销轴传感器（FMx、FMy、FO3x、FO3y），theta关节角(动臂、斗杆、铲斗)，a加速度向量(铲斗，共3维度)，共10维度

    for i in range(length):
        _, sigma = dyn.calc_theta_sigma(distance[i, 0], distance[i, 1], distance[i, 2])
        accelerate[i, :] = dyn.calcAcc(theta[i, :], theta_t[i, :], theta_tt[i, :])
        # print(accelerate[i, :].shape)
        # print(rollOutput[i, :].shape, theta[i, :].shape, sigma.shape, acc.shape)
        FLoadCalc[i, :] = dyn.calc_FLoad_by_FRoll(rollOutput[i, :], theta[i, :], accelerate[i, 0:3])  # 根据销轴传感器计算作业阻力值,
        # input_para[i, :] = np.expand_dims(
        #     np.concatenate([rollOutput[i, :].reshape(-1), theta[i, :].reshape(-1), accelerate[i, 0:3].reshape(-1)]),
        #     axis=0)  # 输入参数向量，rollOutput_销轴传感器（FMx、FMy、FO3x、FO3y），theta关节角(动臂、斗杆、铲斗)，a加速度向量(铲斗，共3维度)，共10维度
        if FLoadCalc[i, 0] < 0:  # 强制力矩大于0
            FLoadCalc[i, 0] = 0
        if FLoadCalc[i, 1] < 0:  # 强制力矩大于0
            FLoadCalc[i, 1] = 0
        if FLoadCalc[i, 2] < 0:  # 强制力矩大于0
            FLoadCalc[i, 2] = 0

    # FLoadCalc[:, 0] = 0.64 * FLoadCalc[:, 0]
    # FLoadCalc[:, 1] = 0.03 * FLoadCalc[:, 1]
    # FLoadCalc[:, 2] = 0.16 * FLoadCalc[:, 2]
    # sigma1 = 50  # 窗口大小
    # order1 = 0  # 多项式阶数
    # Ft = FLoadCalc[:, 0]
    # Fn = FLoadCalc[:, 1]
    # Mr = FLoadCalc[:, 2]
    # smoothed_Ft1 = gaussian_filter1d(Ft, sigma=sigma1, order=order1)
    # smoothed_Fn1 = gaussian_filter1d(Fn, sigma=sigma1, order=order1)
    # smoothed_Mr1 = gaussian_filter1d(Mr, sigma=sigma1, order=order1)
    #
    # sigma2 = 100  # 窗口大小
    # order2 = 0  # 多项式阶数
    # smoothed_Fn2 = gaussian_filter1d(smoothed_Fn1, sigma=sigma2, order=order2)
    # smoothed_Mr2 = gaussian_filter1d(smoothed_Mr1, sigma=sigma2, order=order2)
    # FLoadCalc = np.c_[smoothed_Ft1, smoothed_Fn2, smoothed_Mr2]

    # plt.plot(range(FLoadCalc.shape[0]), FLoadCalc[:, 0], label='Ft')
    # plt.plot(range(FLoadCalc.shape[0]), FLoadCalc[:, 1], label='Fn')
    # plt.plot(range(FLoadCalc.shape[0]), FLoadCalc[:, 2], label='Mr')
    # plt.legend()
    # plt.show()

    # distance 油缸位移3
    xi = np.sum(theta, axis=1).reshape(-1, 1)  # 姿态角1

    # 拼接所有数据
    data_process_ = np.c_[t, distance, hydraulic, rollOutput, xi, accelerate[:, 0:3], FLoadCalc]
    # 保存数据
    # header = "time, 动臂位移, 斗杆位移, 铲斗位移, p_db_d, p_db_x, p_dg_d, p_dg_x, p_cd_d, p_cd_x,
    # Mx, My, O3x, O3y, 姿态角ξ, ax, ay, alpha, Ft, Fn, Mr"
    # np.savetxt(saveFile, data_process_, delimiter=",", header=header, comments='')
    return data_process_


class Get_material_face_fun:
    """
    获取物料表面函数
    """

    def __init__(self, parameters):
        """
        :param parameters: 物料表面函数多项式系数
        """
        self.parameters = parameters

    def matFace(self, x):
        """
        物料拟合曲面
        :param x:
        :return:
        """
        y = np.polyval(self.parameters, x)
        return y

    def calc_d_beta(self, dis, alpha):
        """
        计算挖掘深度d，挖掘前角beta
        :param dis:
        :param alpha:
        :return:
        """
        epsilon = machine_params.get("bucket")["side_bottom_angle"]  # 铲斗侧边与底板夹角（保密参数）
        tra_p = calc_pos_kxi(dis)  # 计算当前位置与姿态
        kxi = tra_p[3] + tra_p[4] + tra_p[5]  # 当前位置的铲斗位姿角
        x = tra_p[0]  # 当前铲斗位置横坐标
        y = tra_p[2]  # 当前铲斗位置纵坐标
        yf = self.matFace(x)  # 根据物料表面函数求出当前位置对应点物料表面坐标
        d = 0.001 * ((yf - y) * sind(90 - alpha))  # 当前时刻挖掘深度
        beta = kxi + alpha + 180 - epsilon  # 计算铲斗前角
        return d, beta

    def calc_alpha(self, dis):
        """
        获取物料堆积角
        :param dis:
        :return:
        """
        tra_p_now = calc_pos_kxi(dis)  # 计算当前时刻位置与姿态
        x_now = tra_p_now[0]  # 当前时刻铲斗斗尖位置横坐标
        interval = 150  # 间隔点为
        x1 = x_now - interval
        x2 = x_now + interval
        y1 = self.matFace(x1)
        y2 = self.matFace(x2)
        slope = - (y1 - y2) / (x1 - x2)  # 求斜率
        alpha = atand(slope)
        return alpha


def calc_pos_kxi(dis):
    """
    根据油缸位移计算铲斗斗尖位置及姿态
    :param dis: 铲斗油缸位移向量
    :return:
    """
    theta1 = 0
    x_db = dis[0]
    x_dg = dis[1]
    x_cd = dis[2]
    # pos前三列储存位置信息，后三列储存角度信息
    k = Kinematic()
    tra = k.kine_qdTowz(theta1, x_db, x_dg, x_cd)  # 计算铲斗斗尖位置及姿态
    return tra


def calc_v(dis_before, dis_now):
    """
    计算铲斗斗尖速度
    :param dis_before: 前一时刻油缸位移向量
    :param dis_now: 当前时刻油缸位移向量
    :return:铲斗斗尖速度
    """
    t_interval_ = 0.01
    tra_p_bef = calc_pos_kxi(dis_before)  # 计算前一时刻位置与姿态
    tra_p_now = calc_pos_kxi(dis_now)  # 计算当前时刻位置与姿态
    x_bef = tra_p_bef[0]  # 前一时刻铲斗位置横坐标
    y_bef = tra_p_bef[2]  # 前一时刻铲斗位置纵坐标
    x_now = tra_p_now[0]  # 当前时刻铲斗位置横坐标
    y_now = tra_p_now[2]  # 当前时刻铲斗位置纵坐标
    v = 0.001 * (math.sqrt((x_now - x_bef) ** 2 + (y_now - y_bef) ** 2)) / t_interval_  # 根据相邻点位置计算速度
    return v


def calc_delta_Vg(dis_before, dis_now, d, beta, w):
    """
    计算累积土壤体积每一时刻增量
    :param dis_before:
    :param dis_now:
    :param d:
    :param beta:
    :param w:
    :return:
    """
    tra_p_bef = calc_pos_kxi(dis_before)  # 计算前一时刻位置与姿态
    tra_p_now = calc_pos_kxi(dis_now)  # 计算当前时刻位置与姿态
    x_bef = tra_p_bef[0]  # 前一时刻铲斗位置横坐标
    y_bef = tra_p_bef[2]  # 前一时刻铲斗位置纵坐标
    x_now = tra_p_now[0]  # 前一时刻铲斗位置横坐标
    y_now = tra_p_bef[2]  # 前一时刻铲斗位置纵坐标
    length_ = 0.001 * (math.sqrt((x_now - x_bef) ** 2 + (y_now - y_bef) ** 2))  # 斗尖位移长度
    S = length_ * sind(abs(beta)) * d
    delta_V = w * S
    return delta_V


def calc_FRS(d, beta, Vg, alpha, delta, phi, rho, c, ca, w):
    """
    用于计算切削阻力FRS
    :param d: 挖掘深度
    :param beta: 挖掘前角
    :param Vg: 累积土壤体积
    :param alpha: 物料堆积角
    :param delta: 土壤与铲斗摩擦角
    :param phi: 土壤内摩擦角
    :param rho: 土壤密度
    :param c: 土壤内聚力系数
    :param ca: 土壤对铲斗附着系数
    :param w: 铲斗宽度
    :return:
    """
    g = 9.8  # 重力加速度
    l_bucket = machine_params.get("bucket")["bottom_length_contact"]  # 铲斗底板长度（保密参数）
    gamma = 90 - (beta + delta + phi) / 2  # 土楔破坏角
    F_c = 0
    F_ca = 0
    W = 0
    G = 0
    L_insert = 0  # 铲斗底板与土壤接触长度
    if d <= 0 and (l_bucket * sind(beta) - d) < 0 and Vg == 0:  # 未插入土壤
        L_insert = 0
        F_ca = 0
        F_c = 0
        W = 0
        G = 0
        # print("1")
    elif 0 < d < l_bucket * sind(beta):  # 铲斗插入土壤，但是底板没有完全插入
        L_insert = (d / sind(beta))
        F_ca = ca * w * L_insert
        F_c = c * w * (d / sind(gamma))
        W = 0.5 * rho * g * w * d * d / (tand(gamma))
        G = rho * g * Vg
        # print('2')
    elif d > 0 and d >= l_bucket * sind(beta):  # 铲斗插入土壤，且底板完全插入
        if beta >= 0:  # 如果切入角度大于等于0
            L_insert = l_bucket
            F_ca = ca * w * L_insert
            F_c = c * w * (d / sind(gamma))
            W = 0.5 * rho * g * w * d * d / (tand(gamma))
            G = rho * g * Vg
            # print('3')
        else:  # 如果切入角度小于0
            L_insert = l_bucket
            F_ca = ca * w * L_insert
            F_c = c * w * (d / sind(gamma))
            W = 0.5 * rho * g * w * d * d / (tand(gamma))
            G = rho * g * Vg
            # print('4')
    elif beta < 0 and d <= 0 and (l_bucket * sind(-beta) + d) > 0 and Vg != 0:  # 铲斗斗尖离开土壤，铲斗底板未完全离开土壤
        L_insert = (l_bucket - (d / sind(beta)))
        F_ca = ca * w * L_insert
        F_c = 0
        W = 0
        G = rho * g * Vg
        # print('5')
    elif d <= 0 and (l_bucket * sind(-beta) + d) <= 0 and Vg != 0:  # 铲斗完全离开土壤
        L_insert = 0
        F_ca = 0
        F_c = 0
        W = 0
        G = rho * g * Vg
        # print('6')
    # F_RS = (-F_ca * cosd(beta + gamma + phi) + (W + G) * sind(alpha + gamma + phi) + F_c * cosd(phi)) / \
    #        (sind(delta + beta + gamma + phi))

    F_RS = ((W + G) * sind(alpha + gamma + phi) + F_c * cosd(phi)) / \
           (sind(delta + beta + gamma + phi))

    return F_RS, L_insert, W


def get_point_cloud_fit(point_cloud_file, point_cloud_position_file):
    """
    获取点云数据，并进行坐标系转换
    :param point_cloud_file:
    :param point_cloud_position_file:
    :return:
    """
    point_cloud_position_distance = np.loadtxt(point_cloud_position_file)  # 位移传感器数据，动臂、斗杆、铲斗
    db_x = np.average(point_cloud_position_distance[:, 13])
    dg_x = np.average(point_cloud_position_distance[:, 14])
    cd_x = np.average(point_cloud_position_distance[:, 15])
    translated_point_cloud = trans_points_cloud_no_color(point_cloud_file, db_x=db_x, dg_x=dg_x, cd_x=cd_x)
    return translated_point_cloud


def get_input_data(preprocessed_data, point_cloud_file, point_cloud_position_file, delta__, phi__, rho__, c__, ca__,
                   volume__, kp__):
    """
    经过计算作业阻力的数据，计算输入参数
    :param preprocessed_data: 计算作业阻力后保存的数据
    :param point_cloud_file: 点云数据
    :param point_cloud_position_file: 获取点云数据时的位移姿态数据
    :param delta__: 物料与铲斗间摩擦角
    :param phi__: 物料内摩擦角
    :param rho__: 物料密度
    :param c__: 物料内聚力系数
    :param ca__: 物料对铲斗的附着系数
    :param volume__: 真实体积
    :param kp__: 挖掘难度
    :return:
    """
    start_time = time.time()
    distance = preprocessed_data[:, 1:4]  # 位移传感器数据，动臂、斗杆、铲斗
    hydraulic = preprocessed_data[:, 4:10]
    rollOutput = preprocessed_data[:, 10:14]
    xi = preprocessed_data[:, 14]
    acc = preprocessed_data[:, 15:18]
    length = len(distance)

    # 获取物理表面拟合函数的参数
    points_Cloud = get_point_cloud_fit(point_cloud_file, point_cloud_position_file)  # 获取点云数据并进行坐标系转换
    para = get_mat_face_func(points_Cloud, degree=4)  # 4次多项式拟合，返回值为四次多项式系数

    # 实例化获取物理表面函数的类
    get_face_fun = Get_material_face_fun(para)

    # 预先分配储存空间
    d__ = np.zeros((length, 1))
    v__ = np.zeros((length, 1))
    beta__ = np.zeros((length, 1))
    Vg__ = np.zeros((length, 1))
    alpha__ = np.zeros((length, 1))
    pos = np.zeros((length, 6))
    mat_face = np.zeros((length, 1))
    w__ = machine_params.get("bucket")["width"]
    epsilon__ = machine_params.get("bucket")["side_bottom_angle"]
    l_bucket__ = machine_params.get("bucket")["bottom_length"]  # 铲斗底板长度（保密参数）
    r__ = machine_params.get("bucket")["back_radius"]  # 铲斗后板半径（保密参数）

    n = 0
    for i in range(length):
        pos[i, :] = calc_pos_kxi(distance[i, :])
        mat_face[i, :] = get_face_fun.matFace(pos[i, 1])
        alpha1 = get_face_fun.calc_alpha(distance[i, :])  # 计算物料表面角度
        d1, _ = get_face_fun.calc_d_beta(distance[i, :], alpha1)  # 计算插入深度
        if i < length - 1:  # 判断是否为最后一个数据点，如果不是，则计算后一个点的物料表面角度和插入深度
            alpha2 = get_face_fun.calc_alpha(distance[i + 1, :])
            d2, _ = get_face_fun.calc_d_beta(distance[i + 1, :], alpha2)
        else:
            alpha2 = get_face_fun.calc_alpha(distance[i, :])
            d2, _ = get_face_fun.calc_d_beta(distance[i, :], alpha2)

        distance[n, :] = distance[i, :]
        alpha__[n, :] = get_face_fun.calc_alpha(distance[n, :])
        d__[n, :], beta__[n, :] = get_face_fun.calc_d_beta(distance[n, :], alpha__[n, :])

        if n == 0:
            v__[n, :] = calc_v(distance[n, :], distance[n, :])
            if d1 >= 0 and d2 >= 0:  # n==0时，只有在插入阶段才按照下式计算，否则为0
                Vg__[n, :] = calc_delta_Vg(distance[n, :], distance[n, :], d__[n, :], beta__[n, :], w__)
            else:
                Vg__[n, :] = 0
        else:
            v__[n, :] = calc_v(distance[n - 1, :], distance[n, :])
            if d1 <= 0 and d2 <= 0:  # 未开始铲掘阶段或铲掘结束阶段
                d_excavator = 0
            else:  # 铲掘阶段
                d_excavator = d__[n, :]
            Vg__[n, :] = calc_delta_Vg(distance[n - 1, :], distance[n, :], d_excavator, beta__[n, :], w__) \
                         + Vg__[n - 1, :]
            # if Vg__[n, :] > 0.02 and d1 < 0:  # 在挖掘完成阶段（铲斗斗尖离开土壤），如果累积土壤体积大于铲斗体积0.02
            #     Vg__[n, :] = 0.02
        n = n + 1

    # distance_ = distance[0:n, :]
    d_ = d__[0:n, :]
    # v_ = v__[0:n, :]
    beta_ = beta__[0:n, :]
    Vg_ = Vg__[0:n, :]
    alpha_ = alpha__[0:n, :]
    delta_ = np.full((n, 1), delta__)
    phi_ = np.full((n, 1), phi__)
    rho_ = np.full((n, 1), rho__)
    c_ = np.full((n, 1), c__)
    ca_ = np.full((n, 1), ca__)
    # k0_ = np.full((n, 1), k0__)
    w_ = np.full((n, 1), w__)
    epsilon_ = np.full((n, 1), epsilon__)
    l_bucket_ = np.full((n, 1), l_bucket__)
    r_ = np.full((n, 1), r__)
    volume_ = np.full((n, 1), volume__)
    kp_ = np.full((n, 1), kp__)

    # input_data = np.c_[d_, v_, beta_, Vg_, alpha_, delta_, phi_, rho_, c_, ca_, k0_, w_, epsilon_]
    # input_data = np.c_[d_, beta_, Vg_, alpha_, delta_, phi_, rho_, c_, ca_, w_, epsilon_]

    input_data = np.c_[d_, beta_, Vg_, alpha_, delta_, phi_, rho_, c_, ca_, w_, epsilon_,
    kp_, distance, xi, v__, acc, pos, mat_face, hydraulic, rho_, w_,
    l_bucket_, r_, volume_, rollOutput]
    end_time = time.time()
    runtime = end_time - start_time
    # header = "d, v, beta, Vg, alpha, delta, phi, rho, c, ca, k0, w, epsilon"
    # np.savetxt(saveFile, input_data, delimiter=",", header=header, comments='')
    return input_data, runtime


def get_phy_out_data(inputData):
    """
    计算铲斗土壤相互作用模型的物理模型输出值
    :param inputData:
    :return:
    """

    d_ = inputData[:, 0]  # 挖掘深度
    # v_ = inputData[:, 1]  # 挖掘速度
    beta_ = inputData[:, 1]  # 挖掘前角
    Vg_ = inputData[:, 2]  # 累积土壤体积
    alpha_ = inputData[:, 3]  # 物理堆积角
    delta_ = inputData[:, 4]  # 铲斗土壤摩擦角
    phi_ = inputData[:, 5]  # 土壤内摩擦角
    rho_ = inputData[:, 6]  # 密度
    c_ = inputData[:, 7]  # 挖掘深度
    ca_ = inputData[:, 8]  # 挖掘深度
    # k0_ = inputData[:, 10]  # 挖掘深度
    w_ = inputData[:, 9]  # 铲斗宽度
    epsilon_ = inputData[:, 10]  # 铲斗底板与侧边缘夹角

    # z1 = 0.08  # 斗尖深度
    length = len(d_)
    # 预先分配储存空间
    Ft = np.zeros((length, 1))
    Fn = np.zeros((length, 1))
    Mr = np.zeros((length, 1))
    F_RS = np.zeros((length, 1))
    # F_P = np.zeros((length, 1))
    # F_SF1 = np.zeros((length, 1))
    # F_ACF1 = np.zeros((length, 1))
    W = np.zeros((length, 1))
    L = np.zeros((length, 1))
    for i in range(length):
        # gamma = 90 - (beta_[i] + delta_[i] + phi_[i]) / 2  # 土楔破坏角
        F_RS[i], L[i], W[i] = calc_FRS(d_[i], beta_[i], Vg_[i], alpha_[i], delta_[i], phi_[i], rho_[i], c_[i],
                                       ca_[i], w_[i])

        # 切向力
        Ft[i] = - F_RS[i] * sind(delta_[i] - epsilon_[i])

        # 法向力
        Fn[i] = F_RS[i] * cosd(epsilon_[i] - delta_[i])

        # 力矩
        Mr[i] = F_RS[i] * L[i] * cosd(delta_[i])

    phy_output_data = np.c_[Ft, Fn, Mr]
    # header = "铲斗土壤Ft, 铲斗土壤Fn, 铲斗土壤Mr, F_RS, L, W"
    # np.savetxt(saveFile, phy_output_data, delimiter=",", header=header, comments='')
    return phy_output_data


def merge_data(input_para_data, phy_output_data_data, resistance_data, saveFile):
    phy_output_data = phy_output_data_data[:, 0:3]  # 土壤相互作用模型计算的铲斗土壤Ft, 铲斗土壤Fn, 铲斗土壤Mr

    t = resistance_data[:, 0]
    resistance_data = resistance_data[:, 18:21]  # 动力学模型计算的铲斗土壤Ft, 铲斗土壤Fn, 铲斗土壤Mr

    data = np.c_[t, input_para_data, phy_output_data, resistance_data]
    header = "t, d, beta, Vg, alpha, delta, phi, rho, c, ca, w, epsilon, " \
             "kp, d_db, d_dg, d_cd, kxi, v, acc_x, acc_y, acc_w, P_x, P_y, P_z, theta1, theta2, theta3, M_z," \
             "p_db_d, p_db_x, p_dg_d, p_dg_x, p_cd_d, p_cd_x, " \
             "rho, w, l_b, r, volume, Mx, My, O3x, O3y, Ft_phy, Fn_phy, Mr_phy, Ft_true, Fn_true, Mr_true"
    np.savetxt(saveFile, data, delimiter=",", header=header, comments='')
    return data


def data_process(original_trajectory_file_path_, original_point_cloud_file_path_,
                 original_point_position_file_, new_dataset_file_path_,
                 file_index, new_data_name, material):
    """
    :param original_trajectory_file_path_: 原始轨迹数据路径
    :param original_point_cloud_file_path_: 原始点云数据路径
    :param original_point_position_file_: 原始点云位置姿态数据路径
    :param new_dataset_file_path_: 保存在新的文件路径
    :param file_index: 文件的索引号
    :param new_data_name: 新文件名
    :param material: 物料种类
    :return:
    """
    all_volume_data = 0.001 * np.loadtxt(open(r'./../OriginalData/original_data/Volume.csv', "rb"), delimiter=",",
                                         skiprows=1, usecols=[0, 1, 2, 3])  # 所有物料的体积数据
    # delta: 物料与铲斗间摩擦角    phi: 物料内摩擦角    rho: 物料密度    c: 物料内聚力系数    ca: 物料对铲斗的附着系数
    delta, phi, rho, c, ca, kp, volume = 30, 30, 2000, 10, 10000, 1, 0
    if material == "sand":  # 沙子的物料参数
        delta, phi, rho, c, ca, kp = 30, 37, 1200, 10, 10, 1  # 已查：phi
        volume = all_volume_data[file_index - 1, 0]
    elif material == "small_stone":  # 细石子的物料参数
        delta, phi, rho, c, ca, kp = 30, 30, 1500, 10, 10, 2  # 已查
        volume = all_volume_data[file_index - 1, 2]
    elif material == "stone":  # 石子的物料参数
        delta, phi, rho, c, ca, kp = 30, 30, 1800, 10, 10, 3  # 已查
        volume = all_volume_data[file_index - 1, 3]
    elif material == "soil":  # 土壤的物料参数
        delta, phi, rho, c, ca, kp = 30, 30, 1300, 10, 10, 1  # 已查：phi
        volume = all_volume_data[file_index - 1, 1]
    else:
        print("物料种类输入错误")

    # 数据预处理，并根据销轴传感器计算作业阻力
    preprocessed_data = data_preprocess_and_calc_resistance(
        original_trajectory_file_path_ + "trajectory_{}.csv".format(file_index))

    # 计算输入参数，preprocess_file为计算作业阻力（根据销轴传感器计算，作为真实值）后的文件，point_cloud_file为点云文件，point_cloud_position_file为获取点云时的油缸姿态位移，saveFile为保存的输入参数文件
    input_data, runtime = get_input_data(
        preprocessed_data=preprocessed_data,
        point_cloud_file=original_point_cloud_file_path_ + "Point_cloud_{}.ply".format(file_index),
        point_cloud_position_file=original_point_position_file_ + 'Point_cloud_position_{}.txt'.format(file_index),
        delta__=delta, phi__=phi, rho__=rho, c__=c, ca__=ca, volume__=volume, kp__=kp)

    # 计算铲斗土壤相互作用模型的输出，inputData为输入参数，saveFile为保存的通过铲斗土壤相互作用模型计算的作业阻力文件Ft, Fn, Mr
    phy_out_data = get_phy_out_data(inputData=input_data)

    # 数据组成[t, d, v, beta, Vg, alpha, delta, phi, rho, c, ca, k0, w, epsilon, Ft_, Fn_, Mr_, Ft, Fn, Mr]，
    # Ft_, Fn_, Mr_为铲斗土壤相互作用模型计算值，Ft, Fn, Mr为真实值（销轴传感器获取）
    merge_data(input_para_data=input_data, phy_output_data_data=phy_out_data, resistance_data=preprocessed_data,
               saveFile=new_dataset_file_path_ + "{}_{}.csv".format(new_data_name, file_index))
    return runtime


def main():
    # 细沙
    original_sand_trajectory_file_path = \
        "./../OriginalData/original_data/Sand/Sand_Trajectory/"  # 轨迹数据文件路径
    original_sand_point_cloud_file_path = \
        "./../OriginalData/original_data/Sand/Sand_Point/"  # 点云数据文件路径
    original_sand_point_cloud_position_file_path = \
        "./../OriginalData/original_data/Sand/Sand_Point_Position/"  # 点云位置文件路径
    new_sand_dataset_file_path = \
        "./../OriginalData/All_material_data_excavation_3Classes_P_filter" \
        "/Preprocessed_data/Sand_data/"  # 处理后的数据文件保存路径

    # 小石子
    original_small_stone_trajectory_file_path = \
        "./../OriginalData/original_data/Small_Stone/Small_Stone_Trajectory/"  # 轨迹数据文件路径
    original_small_stone_point_cloud_file_path = \
        "./../OriginalData/original_data/Small_Stone/Small_Stone_Point/"  # 点云数据文件路径
    original_small_stone_point_cloud_position_file_path = \
        "./../OriginalData/original_data/Small_Stone/Small_Stone_Point_Position/"  # 点云位置文件路径
    new_small_stone_dataset_file_path = \
        "./../OriginalData/All_material_data_excavation_3Classes_P_filter" \
        "/Preprocessed_data/Small_stone_data/"  # 处理后的数据文件保存路径

    # 石子
    original_stone_trajectory_file_path = \
        "./../OriginalData/original_data/Stone/Stone_Trajectory/"  # 轨迹数据文件路径
    original_stone_point_cloud_file_path = \
        "./../OriginalData/original_data/Stone/Stone_Point/"  # 点云数据文件路径
    original_stone_point_cloud_position_file_path = \
        "./../OriginalData/original_data/Stone/Stone_Point_Position/"  # 点云位置文件路径
    new_stone_dataset_file_path = \
        "./../OriginalData/All_material_data_excavation_3Classes_P_filter" \
        "/Preprocessed_data/Stone_data/"  # 处理后的数据文件保存路径

    # 土壤
    original_soil_trajectory_file_path = \
        "./../OriginalData/original_data/Soil/Soil_Trajectory/"  # 轨迹数据文件路径
    original_soil_point_cloud_file_path = \
        "./../OriginalData/original_data/Soil/Soil_Point/"  # 点云数据文件路径
    original_soil_point_cloud_position_file_path = \
        "./../OriginalData/original_data/Soil/Soil_Point_Position/"  # 点云位置文件路径
    new_soil_dataset_file_path = \
        "./../OriginalData/All_material_data_excavation_3Classes_P_filter" \
        "/Preprocessed_data/Soil_data/"  # 处理后的数据文件保存路径
    runtime_phy = []
    start_time = time.time()
    for file_i in range(150):
        # data_process(original_trajectory_file_path_="./../OriginalData/original_data/Data_small_stone_0704/Trajectory/",
        #              original_point_cloud_file_path_="./../OriginalData/original_data/Data_small_stone_0704/Point/",
        #              original_point_position_file_="./../OriginalData/original_data/Data_small_stone_0704"
        #                                            "/Point_Position/",
        #              new_dataset_file_path_="./../OriginalData/All_material_data_excavation_3Classes_P_filter"
        #                                     "/Preprocessed_data/Data_small_stone_0704/",
        #              file_index=file_i + 1, new_data_name="small_stone_data", material="small_stone")

        # 细沙
        runtime_sand = data_process(original_sand_trajectory_file_path, original_sand_point_cloud_file_path,
                                    original_sand_point_cloud_position_file_path, new_sand_dataset_file_path,
                                    file_index=file_i + 1, new_data_name="sand_data", material="sand")
        # 小石子
        runtime_small_stone = data_process(original_small_stone_trajectory_file_path,
                                           original_small_stone_point_cloud_file_path,
                                           original_small_stone_point_cloud_position_file_path,
                                           new_small_stone_dataset_file_path,
                                           file_index=file_i + 1, new_data_name="small_stone_data",
                                           material="small_stone")
        # 石子
        runtime_stone = data_process(original_stone_trajectory_file_path, original_stone_point_cloud_file_path,
                                     original_stone_point_cloud_position_file_path, new_stone_dataset_file_path,
                                     file_index=file_i + 1, new_data_name="stone_data", material="stone")

        # 土壤
        runtime_soil = data_process(original_soil_trajectory_file_path, original_soil_point_cloud_file_path,
                                    original_soil_point_cloud_position_file_path, new_soil_dataset_file_path,
                                    file_index=file_i + 1, new_data_name="soil_data", material="soil")
        runtime_phy.append((runtime_sand + runtime_small_stone + runtime_stone + runtime_soil) / 4)
    end_time = time.time()
    average_time = (end_time - start_time) / 600
    print(f"average time:{average_time}")
    print(f"average runtime phy_model:{np.mean(runtime_phy)}")


if __name__ == '__main__':
    main()
