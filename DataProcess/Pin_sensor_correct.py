"""Pin-sensor force correction based on the working-equipment dynamics.

The implementation contains the confidential geometry and dynamic parameters of the test
excavator and is stored encrypted in machine_params.enc (see machine_params.py).
"""
import machine_params

machine_params.exec_module("DataProcess/Pin_sensor_correct.py", globals())
