import math
import torch
import numpy as np
"""
根据角度值计算三角函数和反三角函数
"""


def cosd(angle):
    # y = math.cos(math.radians(angle))
    # y = torch.cos(torch.deg2rad(angle))
    y = np.cos(np.radians(angle))
    return y


def sind(angle):
    # y = math.sin(math.radians(angle))
    # y = torch.sin(torch.deg2rad(angle))
    y = np.sin(np.radians(angle))
    return y


def tand(angle):
    # y = math.tan(math.radians(angle))
    # y = torch.tan(torch.deg2rad(angle))
    y = np.tan(np.radians(angle))
    return y


def cotd(angle):
    # y = 1 / (math.tan(math.radians(angle)))
    # y = 1 / torch.tan(torch.deg2rad(angle))
    y = 1 / np.tan(np.radians(angle))
    return y


def acosd(angle):
    # y = math.degrees(math.acos(angle))
    # y = torch.rad2deg(torch.acos(angle))
    y = np.degrees(np.arccos(angle))
    return y


def asind(angle):
    # y = math.degrees(math.asin(angle))
    # y = torch.rad2deg(torch.asin(angle))
    y = np.degrees(np.arcsin(angle))
    return y


def atand(angle):
    # y = math.degrees(math.atan(angle))
    # y = torch.rad2deg(torch.atan(angle))
    y = np.degrees(np.arctan(angle))
    return y
