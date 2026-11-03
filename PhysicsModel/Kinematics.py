"""Excavator kinematics: conversion between bucket-tip pose and cylinder displacements.

The implementation contains the confidential geometry and dynamic parameters of the test
excavator and is stored encrypted in machine_params.enc (see machine_params.py).
"""
import machine_params

machine_params.exec_module("PhysicsModel/Kinematics.py", globals())
