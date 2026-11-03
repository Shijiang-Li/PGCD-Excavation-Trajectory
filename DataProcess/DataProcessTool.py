import numpy as np
from scipy.signal import butter
import scipy.signal as signal
from trigonometric_function import *


def normalize_data(input_data, up_limit=5, down_limit=-5):
    """
    限定输出值范围
    :param input_data: 输入数据
    :param up_limit: 上限值
    :param down_limit: 下限值
    :return: 输出数据
    """
    input_data[input_data > up_limit] = up_limit  # 替换NumPy数组中大于某个值的所有元素
    input_data[input_data < down_limit] = down_limit  # 替换NumPy数组中小于某个值的所有元素
    return input_data


def filterHAMPEL(data, k):
    """
    H A M P E L filter是在滚动窗口下识别异常值并进行使用窗口的中位数进行替换的方法。
    :param data: 输入数据
    :param k: 半个窗口宽度为k，整个窗口的宽度为2k+1（包含中心的那个元素）
    :return:
    """
    data_length = data.shape[0] - 1
    nSigma = 3
    iLo = np.array([index - k for index in range(0, data_length + 1)])
    iHi = np.array([index + k for index in range(0, data_length + 1)])
    iLo[iLo < 0] = 0
    iHi[iHi > data_length] = data_length
    x_mad = []
    x_median = []
    for index in range(data_length + 1):
        w = data[iLo[index]:iHi[index] + 1]
        med_j = np.median(w)
        mad = np.median(np.abs(w - med_j))
        x_mad.append(mad)
        x_median.append(med_j)
    x_mad = np.array(x_mad)
    x_median = np.array(x_median)
    scale = 1.4826  # 缩放
    xSigma = scale * x_mad
    xi = ~(np.abs(data - x_median) <= nSigma * xSigma)  # 找出离群点（即超过nSigma个标准差）

    # 将离群点替换为中为数值
    xf = data.copy()
    xf[xi] = x_median[xi]
    return xf


def removeAndFillOutlier(distanceData, hydraulicData, rollOutputData, k=3):
    db_dis = filterHAMPEL(distanceData[:, 0], k)
    dg_dis = filterHAMPEL(distanceData[:, 1], k)
    cd_dis = filterHAMPEL(distanceData[:, 2], k)
    distanceOut = np.c_[db_dis, dg_dis, cd_dis]

    db_pressure_big = filterHAMPEL(hydraulicData[:, 0], k)
    db_pressure_small = filterHAMPEL(hydraulicData[:, 1], k)
    dg_pressure_big = filterHAMPEL(hydraulicData[:, 2], k)
    dg_pressure_small = filterHAMPEL(hydraulicData[:, 3], k)
    cd_pressure_big = filterHAMPEL(hydraulicData[:, 4], k)
    cd_pressure_small = filterHAMPEL(hydraulicData[:, 5], k)
    hydraulicOut = np.c_[
        db_pressure_big, db_pressure_small, dg_pressure_big, dg_pressure_small, cd_pressure_big, cd_pressure_small]

    Fx2_M = filterHAMPEL(rollOutputData[:, 0], k)
    Fy2_M = filterHAMPEL(rollOutputData[:, 1], k)
    Fx1_O3 = filterHAMPEL(rollOutputData[:, 2], k)
    Fy1_O3 = filterHAMPEL(rollOutputData[:, 3], k)
    rollOutputOut = np.c_[Fx2_M, Fy2_M, Fx1_O3, Fy1_O3]
    return distanceOut, hydraulicOut, rollOutputOut


def MeanMedianValuesFilter(Data, N):
    """
    A、名称：中位值平均滤波法（又称防脉冲干扰平均滤波法）
    B、方法：采一组队列去掉最大值和最小值后取平均值，相当于“中位值滤波法”+“算术平均滤波法”。连续采样N个数据，去掉一个最大值和一个最小值，然后计算N-2个数据的算术平均值。	N值的选取：3-14。
    C、优点：融合了“中位值滤波法”+“算术平均滤波法”两种滤波法的优点。对于偶然出现的脉冲性干扰，可消除由其所引起的采样值偏差。对周期干扰有良好的抑制作用。平滑度高，适于高频振荡的系统。
    D、缺点：计算速度较慢，和算术平均滤波法一样，比较浪费RAM。
    :param Data: 输入数据
    :param N: 连续采样数据量
    :return: 滤波后的数据
    """
    ReturnData = []
    StageList = []

    for index, Value in enumerate(Data):
        StageList.append(Value)
        if (index + 1) % N == 0:
            StageList.sort()  # 排序
            StageList.pop()
            StageList.pop(0)  # 删除最大值与最小值
            ReturnData += [sum(StageList) / len(StageList)] * N
            StageList.clear()  # 清空列表

    # 剩余数据处理
    if len(StageList) != 0:
        if len(StageList) == 1:  # 剩余一个的处理
            ReturnData.append(StageList[0])
        elif len(StageList) == 2:  # 剩余两个的处理
            ReturnData += [(StageList[0] + StageList[1]) / 2] * 2
        else:  # 剩余两个以上的处理
            Residue = len(StageList)
            StageList.sort()
            StageList.pop()
            StageList.pop(0)
            ReturnData += [sum(StageList) / len(StageList)] * Residue

    return ReturnData


def MeanMedianValuesFilterData(distanceData, hydraulicData, rollOutputData, N_f=4):
    """
    利用均值中位值滤波器进行滤波
    :param distanceData: 位移数据
    :param hydraulicData: 液压数据
    :param rollOutputData: 销轴传感器数据
    :param N_f:连续采样数据量
    :return:
    """
    db_dis = MeanMedianValuesFilter(distanceData[:, 0], N_f)
    dg_dis = MeanMedianValuesFilter(distanceData[:, 1], N_f)
    cd_dis = MeanMedianValuesFilter(distanceData[:, 2], N_f)
    distanceOut = np.c_[db_dis, dg_dis, cd_dis]

    db_pressure_big = MeanMedianValuesFilter(hydraulicData[:, 0], N_f)
    db_pressure_small = MeanMedianValuesFilter(hydraulicData[:, 1], N_f)
    dg_pressure_big = MeanMedianValuesFilter(hydraulicData[:, 2], N_f)
    dg_pressure_small = MeanMedianValuesFilter(hydraulicData[:, 3], N_f)
    cd_pressure_big = MeanMedianValuesFilter(hydraulicData[:, 4], N_f)
    cd_pressure_small = MeanMedianValuesFilter(hydraulicData[:, 5], N_f)
    hydraulicOut = np.c_[
        db_pressure_big, db_pressure_small, dg_pressure_big, dg_pressure_small, cd_pressure_big, cd_pressure_small]

    Fx2_M = MeanMedianValuesFilter(rollOutputData[:, 0], N_f)
    Fy2_M = MeanMedianValuesFilter(rollOutputData[:, 1], N_f)
    Fx1_O3 = MeanMedianValuesFilter(rollOutputData[:, 2], N_f)
    Fy1_O3 = MeanMedianValuesFilter(rollOutputData[:, 3], N_f)
    rollOutputOut = np.c_[Fx2_M, Fy2_M, Fx1_O3, Fy1_O3]
    return distanceOut, hydraulicOut, rollOutputOut


def low_pass_filter(data, cutoff_freq=50, order=2):
    """
    巴特沃斯低通滤波器
    :param data:
    :param cutoff_freq:
    :param order:
    :return:
    """
    fs = 100
    normalized_cutoff = cutoff_freq / (0.5 * fs)
    [b, a_] = butter(order, normalized_cutoff, btype='low', analog=False)
    filtered_signal = signal.lfilter(b, a_, data)
    return filtered_signal


def lossPassFilter(distanceData, hydraulicData, rollOutputData, cutoff_freq=50, order=4):
    db_dis = low_pass_filter(distanceData[:, 0], cutoff_freq, order)
    dg_dis = low_pass_filter(distanceData[:, 1], cutoff_freq, order)
    cd_dis = low_pass_filter(distanceData[:, 2], cutoff_freq, order)
    distanceOut = np.c_[db_dis, dg_dis, cd_dis]

    db_pressure_big = low_pass_filter(hydraulicData[:, 0], cutoff_freq, order)
    db_pressure_small = low_pass_filter(hydraulicData[:, 1], cutoff_freq, order)
    dg_pressure_big = low_pass_filter(hydraulicData[:, 2], cutoff_freq, order)
    dg_pressure_small = low_pass_filter(hydraulicData[:, 3], cutoff_freq, order)
    cd_pressure_big = low_pass_filter(hydraulicData[:, 4], cutoff_freq, order)
    cd_pressure_small = low_pass_filter(hydraulicData[:, 5], cutoff_freq, order)
    hydraulicOut = np.c_[
        db_pressure_big, db_pressure_small, dg_pressure_big, dg_pressure_small, cd_pressure_big, cd_pressure_small]

    Fx2_M = low_pass_filter(rollOutputData[:, 0], cutoff_freq, order)
    Fy2_M = low_pass_filter(rollOutputData[:, 1], cutoff_freq, order)
    Fx1_O3 = low_pass_filter(rollOutputData[:, 2], cutoff_freq, order)
    Fy1_O3 = low_pass_filter(rollOutputData[:, 3], cutoff_freq, order)

    rollOutputOut = np.c_[Fx2_M, Fy2_M, Fx1_O3, Fy1_O3]
    return distanceOut, hydraulicOut, rollOutputOut


def scipyFilter(distanceData, hydraulicData, rollOutputData, win_length=53, k=3):
    """
    利用scipy自带滤波器进行滤波
    :param distanceData: 位移数据
    :param hydraulicData: 液压数据
    :param rollOutputData: 销轴传感器数据
    :param win_length:滤波窗口长度
    :param k:对window_length内的数据进行k阶多项式拟合
    :return:
    """
    db_dis = signal.savgol_filter(distanceData[:, 0], win_length, k)
    dg_dis = signal.savgol_filter(distanceData[:, 1], win_length, k)
    cd_dis = signal.savgol_filter(distanceData[:, 2], win_length, k)
    distanceOut = np.c_[db_dis, dg_dis, cd_dis]

    db_pressure_big = signal.savgol_filter(hydraulicData[:, 0], win_length, k)
    db_pressure_small = signal.savgol_filter(hydraulicData[:, 1], win_length, k)
    dg_pressure_big = signal.savgol_filter(hydraulicData[:, 2], win_length, k)
    dg_pressure_small = signal.savgol_filter(hydraulicData[:, 3], win_length, k)
    cd_pressure_big = signal.savgol_filter(hydraulicData[:, 4], win_length, k)
    cd_pressure_small = signal.savgol_filter(hydraulicData[:, 5], win_length, k)
    hydraulicOut = np.c_[
        db_pressure_big, db_pressure_small, dg_pressure_big, dg_pressure_small, cd_pressure_big, cd_pressure_small]

    Fx2_M = signal.savgol_filter(rollOutputData[:, 0], win_length, k)
    Fy2_M = signal.savgol_filter(rollOutputData[:, 1], win_length, k)
    Fx1_O3 = signal.savgol_filter(rollOutputData[:, 2], win_length, k)
    Fy1_O3 = signal.savgol_filter(rollOutputData[:, 3], win_length, k)

    rollOutputOut = np.c_[Fx2_M, Fy2_M, Fx1_O3, Fy1_O3]
    return distanceOut, hydraulicOut, rollOutputOut


def thetaAccFilter(acc, win_length=53, k=3):
    theta1Acc = signal.savgol_filter(acc[:, 0], win_length, k)
    theta2Acc = signal.savgol_filter(acc[:, 1], win_length, k)
    theta3Acc = signal.savgol_filter(acc[:, 2], win_length, k)
    # acc3x = signal.savgol_filter(acc[:, 3], win_length, k)
    # acc3y = signal.savgol_filter(acc[:, 4], win_length, k)
    # acc3e = signal.savgol_filter(acc[:, 5], win_length, k)
    # acc2x = signal.savgol_filter(acc[:, 6], win_length, k)
    # acc2y = signal.savgol_filter(acc[:, 7], win_length, k)
    # acc2e = signal.savgol_filter(acc[:, 8], win_length, k)
    acc_ = np.c_[theta1Acc, theta2Acc, theta3Acc]
    return acc_


def average_downSampling(data, factor):
    """
    平均值下采样
    :param data: 输入数据
    :param factor:下采样因子
    :return:下采样后的数据
    """
    downSampled_data = np.mean(data.reshape(-1, factor), axis=1)
    return downSampled_data


def downSampling(distanceData, hydraulicData, rollOutputData, factor=2):
    """
    利用scipy自带滤波器进行滤波
    :param distanceData: 位移数据
    :param hydraulicData: 液压数据
    :param rollOutputData: 销轴传感器数据
    :param factor: 下采样因子
    :return:
    """
    db_dis = average_downSampling(distanceData[:, 0], factor)
    dg_dis = average_downSampling(distanceData[:, 1], factor)
    cd_dis = average_downSampling(distanceData[:, 2], factor)
    distanceOut = np.c_[db_dis, dg_dis, cd_dis]

    db_pressure_big = average_downSampling(hydraulicData[:, 0], factor)
    db_pressure_small = average_downSampling(hydraulicData[:, 1], factor)
    dg_pressure_big = average_downSampling(hydraulicData[:, 2], factor)
    dg_pressure_small = average_downSampling(hydraulicData[:, 3], factor)
    cd_pressure_big = average_downSampling(hydraulicData[:, 4], factor)
    cd_pressure_small = average_downSampling(hydraulicData[:, 5], factor)
    hydraulicOut = np.c_[
        db_pressure_big, db_pressure_small, dg_pressure_big, dg_pressure_small, cd_pressure_big, cd_pressure_small]

    Fx2_M = average_downSampling(rollOutputData[:, 0], factor)
    Fy2_M = average_downSampling(rollOutputData[:, 1], factor)
    Fx1_O3 = average_downSampling(rollOutputData[:, 2], factor)
    Fy1_O3 = average_downSampling(rollOutputData[:, 3], factor)

    rollOutputOut = np.c_[Fx2_M, Fy2_M, Fx1_O3, Fy1_O3]
    return distanceOut, hydraulicOut, rollOutputOut


def calculateHydraulicForce(hydraulicData):
    """
    根据压强数据计算油缸输出力
    :param hydraulicData:油缸大小腔压强数据
    :return:油缸输出力
    """
    d_DBT = 0.05  # 动臂油缸体内径
    d_DGT = 0.05  # 斗杆油缸体内径
    d_CDT = 0.050  # 铲斗油缸体内径
    d_DBG = 0.040  # 动臂油缸杆内径
    d_DGG = 0.040  # 斗杆油缸杆内径
    d_CDG = 0.040  # 铲斗油缸杆内径
    s_DB = math.pi * (d_DBT ** 2 - d_DBG ** 2) / 4  # 动臂油缸活塞有效面积
    s_DG = math.pi * (d_DGT ** 2 - d_DGG ** 2) / 4  # 斗杆油缸活塞有效面积
    s_CD = math.pi * (d_CDT ** 2 - d_CDG ** 2) / 4  # 铲斗油缸活塞有效面积
    F_DB = (hydraulicData[:, 0] - hydraulicData[:, 1]) * s_DB * 1e6  # 动臂油缸驱动力
    F_DG = (hydraulicData[:, 2] - hydraulicData[:, 3]) * s_DG * 1e6  # 斗杆油缸驱动力
    F_CD = (hydraulicData[:, 4] - hydraulicData[:, 5]) * s_CD * 1e6  # 铲斗油缸驱动力
    # F_DB = (hydraulicData[:, 0] - hydraulicData[:, 1]) * s_DB  # 动臂油缸驱动力
    # F_DG = (hydraulicData[:, 2] - hydraulicData[:, 3]) * s_DG  # 斗杆油缸驱动力
    # F_CD = (hydraulicData[:, 4] - hydraulicData[:, 5]) * s_CD  # 铲斗油缸驱动力
    drivingForce = np.array([F_DB, F_DG, F_CD]).T  # 油缸驱动力矩阵
    return drivingForce


def calc_FLoad_by_gravity(theta_, m):
    l4 = 0.37
    # 当力的作用点在铲斗斗尖时
    lO3Gm = l4
    gamma = 0
    theta2 = theta_[0]
    theta3 = theta_[1]
    theta4 = theta_[2]
    Ft = - 9.8 * m * cosd(theta2 + theta3 + theta4)
    Fn = 9.8 * m * sind(theta2 + theta3 + theta4)

    # Mr = - 9.8 * m * sind(theta2 + theta3 + theta4) * lO3Gm * sind(gamma) - 9.8 * m * cosd(
    #     theta2 + theta3 + theta4) * lO3Gm * cosd(gamma)
    Mr = 9.8 * m * sind(theta2 + theta3 + theta4) * lO3Gm * sind(gamma) + 9.8 * m * cosd(
        theta2 + theta3 + theta4) * lO3Gm * cosd(gamma)
    FLoad = np.array([Ft, Fn, Mr])
    return FLoad


def calc_FLoad_by_force(theta_, m_force):
    l4 = 0.37
    # 当力的作用点在铲斗斗尖时
    # lO3Gm = l4
    # gamma = 0
    theta2 = theta_[0]
    theta3 = theta_[1]
    theta4 = theta_[2]
    Ft = 9.8 * m_force * sind(180 + theta2 + theta3 + theta4)
    Fn = 9.8 * m_force * cosd(180 + (theta2 + theta3 + theta4))

    # Ft = - 9.8 * m_force * cosd(theta2 + theta3 + theta4 + 90)
    # Fn = 9.8 * m_force * sind(theta2 + theta3 + theta4 + 90)
    # Mr = - 9.8 * m * sind(theta2 + theta3 + theta4) * lO3Gm * sind(gamma) - 9.8 * m * cosd(
    #     theta2 + theta3 + theta4) * lO3Gm * cosd(gamma)
    # Mr = Ft * lO3Gm
    JO3O4d = 90
    # Mr = - (Ft * lO3Gm + 9.8 * m_force * 0.05 * sind(JO3O4d - (180 + theta2 + theta3 + theta4)))
    Mr = (9.8 * m_force * 0.2 * sind(JO3O4d - (180 + theta2 + theta3 + theta4)))
    FLoad = np.array([Ft, Fn, Mr])
    return FLoad
