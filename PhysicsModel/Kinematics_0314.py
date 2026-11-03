"""Excavator kinematics with the calibration of the field-test sensors.

The implementation contains the confidential geometry and dynamic parameters of the test
excavator and is stored encrypted in machine_params.enc (see machine_params.py).
"""
import machine_params

machine_params.exec_module("PhysicsModel/Kinematics_0314.py", globals())
