# SIM-DAC switch
# sub module化
# ファイル名に測定の詳細(温度、磁場、コンタクト、電流)
#　温度読み出し

import labrad
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.io as sio
from matplotlib.ticker import ScalarFormatter

import time
import math
import os

import slackweb
import requests

cxn = labrad.connect()
cxn_2 = labrad.connect()
cxn_3 = labrad.connect()

tcdel = 3.0

ramp_step = 1000
ramp_delay = 2500

adc_channel_0 = 0
adc_channel_1 = 1
adc_channel_2 = 2
adc_channel_3 = 3

# Initialization

DV = cxn.data_vault

# DAC = cxn.dac_adc
# DAC.select_device()
sim = cxn.sim900
sim.select_device()

LA1 = cxn.sr860   #  source
LA1.select_device('sarachick GPIB Bus - GPIB0::6::INSTR') 
LA2 = cxn_2.sr860
LA2.select_device('sarachick GPIB Bus - GPIB0::5::INSTR')
LA3 = cxn_3.sr860
LA3.select_device('sarachick GPIB Bus - GPIB0::4::INSTR') 

#MG = cxn.ami_430
#MG.select_device()
MGz = cxn.ami_430
MGz.select_device('sarachick_serial_server - COM6')
MGy = cxn_2.ami_430
MGy.select_device('sarachick_serial_server - COM7')
MGx = cxn_3.ami_430
MGx.select_device('sarachick_serial_server - COM8')


%reload_ext autoreload
%autoreload 2
import mes
# output
file_path = "WG8_VM"
file_name = "WG8_VM"
#voltage_source = "DAC"


amplitude = 2.0
frequency = 17.7777
gate_gain = 1.0
#278Hz

voltage_gain = 1.0

dc_channel_0 = 0
dc_channel_1 = 1# gate voltage swept

tcdel = 3.0
time_constant_1 = LA1.time_constant()
sensitivity_1 = LA1.sensitivity()
time_constant_2 = LA2.time_constant()
sensitivity_2 = LA2.sensitivity()
time_constant_3 = LA3.time_constant()
sensitivity_3 = LA3.sensitivity()


def main():
    scan_R_vs_Vg_Bz(
        file_path = file_path,
        voltage_source = 'SIM',
        voltage_channel = 3,
        amplitude=0.01,
        frequency=17.777,
        gate_gain=1.0,
        meas_voltage_gain=1.0,
        bz_range=[0.0, 0.0],
        vg_range=[-1.0, 1.0],
        number_of_bz_lines=1,
        number_of_vg_points=200,
        wait_time=wait_time,
        note="misc",
    )


if __name__ == '__main__':
    main()