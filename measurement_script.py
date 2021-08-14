import labrad
import numpy as np
from numpy import linalg as LA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from heapq import heappush, nlargest
import scipy.io as sio
from matplotlib.ticker import ScalarFormatter
import datetime
import time
import math
import os
import inspect
import sys
from tqdm import tqdm

import slackweb
import requests

meas_details_path = "C://Users//Lab//Young Lab Dropbox//Bruefors//Data//vault//Yu_Bruefors//meas_details//"

sns.set(
    "talk",
    "whitegrid",
    "dark",
    font_scale=1.2,
    rc={"lines.linewidth": 2, "grid.linestyle": "--"},
)

cxn = labrad.connect()
cxn_2 = labrad.connect()
cxn_3 = labrad.connect()

tcdel = 3.0

ramp_step = 1000
ramp_delay = 2500

adc_channel_0 = 4
adc_channel_1 = 5
adc_channel_2 = 6
adc_channel_3 = 7

# Initialization

DV = cxn.data_vault

sim = cxn.sim900
sim.select_device()

LA1 = cxn.sr860  #  source
LA1.select_device("sarachick GPIB Bus - GPIB0::6::INSTR")
LA2 = cxn_2.sr860
LA2.select_device("sarachick GPIB Bus - GPIB0::5::INSTR")
LA3 = cxn_3.sr860
LA3.select_device("sarachick GPIB Bus - GPIB0::4::INSTR")

MGz = cxn.ami_430
MGz.select_device("sarachick_serial_server - COM6")
MGy = cxn_2.ami_430
MGy.select_device("sarachick_serial_server - COM7")
MGx = cxn_3.ami_430
MGx.select_device("sarachick_serial_server - COM8")

LAs = [LA1, LA2, LA3]

time_constant_1 = LA1.time_constant()
sensitivity_1 = LA1.sensitivity()
time_constant_2 = LA2.time_constant()
sensitivity_2 = LA2.sensitivity()
time_constant_3 = LA3.time_constant()
sensitivity_3 = LA3.sensitivity()

wait_time = 1e6 * tcdel * time_constant_1

##########################################################################################

# Euler ZYX convention: rotate first Z, then Y, lastly X
# Given magnetic vectors a=(Bx,By,Bz) and b=(Bx2,By2,Bz2), output ZYX rotation angles to transform the normalized a to b


def find_vt_vb(p0, n0, c_delta):  # input p0, n0, return vt, vb
    return 0.5 * (n0 + p0) / (1.0 + c_delta), 0.5 * (n0 - p0) / (1.0 - c_delta)


def calc_rotation_matrix(a1, a2, a3, b1, b2, b3):

    norm1 = np.sqrt(a1 ** 2 + a2 ** 2 + a3 ** 2)
    norm2 = np.sqrt(b1 ** 2 + b2 ** 2 + b3 ** 2)
    a1, a2, a3 = a1 / norm1, a2 / norm1, a3 / norm1
    b1, b2, b3 = b1 / norm2, b2 / norm2, b3 / norm2
    v1 = a2 * b3 - a3 * b2
    v2 = a3 * b1 - a1 * b3
    v3 = a1 * b2 - a2 * b1
    v = np.mat([v1, v2, v3])
    s = np.sqrt(v[0, 0] ** 2 + v[0, 1] ** 2 + v[0, 2] ** 2)
    c = a1 * b1 + a2 * b2 + a3 * b3
    vx = np.mat([[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]])
    I = np.mat([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    return I + vx + vx * vx / (1 + c)


def rotate_vector(R, x, y, z):
    v = np.mat([x, y, z])
    new = np.array(R * v.T)
    return new[0], new[1], new[2]


def get_meas_parameters(offset=None):
    """
    Get a dictionary of paramteres of the function.

    Credit : https://tottoto.net/python3-get-args-of-current-function/

    Parameters
    ----------
    offset : int
        default value is None

    Return
    ------
    dictionary
        The dictionary includes pairs of paremeter's name and the corresponding values.
    """
    parent_frame = inspect.currentframe().f_back
    info = inspect.getargvalues(parent_frame)
    return {key: info.locals[key] for key in info.args[offset:]}


def set_lockin_parameters(amplitude=0.01, frequency=17.777):
    """
    Initialize the lockin amp for voltage source.

    Parameters
    ----------
    amplitude : int or float
        The ampitude value of the lockin.
    frequency : int or float

    Returns
    -------
    None
    """
    LA1.sine_out_amplitude(amplitude)
    LA1.frequency(frequency)
    time.sleep(1)
    print "\r", "parameters set done",


def create_file(DV, file_path, scan_name, scan_var, meas_var):
    """
    Create a measurment file.

    Parameters
    ----------
    DV : object
    file_path : string
    scan_name : string
    scan_var : list or tuple
    meas_var : list or tuple

    Returns
    -------
    int
        The file number

    """
    DV.cd("")
    try:
        DV.mkdir(file_path)
        DV.cd(file_path)
    except Exception:
        DV.cd(file_path)

    file_name = file_path + "_" + scan_name
    dv_file = DV.new(file_name, scan_var, meas_var)
    print "\r", "new file created, file numer: ", int(dv_file[1][0:5])

    return int(dv_file[1][0:5])


def write_meas_parameters(
    DV, file_path, file_number, date, scan_name, meas_parameters, amplitude, sensitivity
):

    if not os.path.isfile(meas_details_path + file_path + ".txt"):
        with open(meas_details_path + file_path + ".txt", "w+") as f:
            pass
    with open(meas_details_path + file_path + ".txt", "a") as f:
        f.write("========" + "\n")
        f.write(
            "file_number: "
            + str(file_number)
            + "\n"
            + "date: "
            + str(date)
            + "\n"
            + "measurement:"
            + str(scan_name)
            + "\n"
        )
        for k, v in sorted(meas_parameters.items()):
            print (k, v)
            f.write(str(k) + ": " + str(v) + "\n")
            DV.add_parameter(str(k), str(v))

        for i, LA in enumerate(LAs):
            tc = LA.time_constant()
            sens = LA.sensitivity()
            f.write("time_constant_" + str(i) + " : " + str(tc) + "\n")
            f.write("sensitivity_" + str(i) + " : " + str(sens) + "\n")
            DV.add_parameter("time_constant_" + str(i), tc)
            DV.add_parameter("sensitivity_" + str(i), sens)


def write_meas_parameters_end(date1, date2, file_path):

    with open(meas_details_path + file_path + ".txt", "a") as f:
        f.write(
            "end date: "
            + str(date2)
            + "\n"
            + "total time: "
            + str(date2 - date1)
            + "\n"
        )


def get_variables():
    variables = [DV.variables()[0][i][0] for i in range(len(DV.variables()[0]))] + [
        DV.variables()[1][i][0] for i in range(len(DV.variables()[1]))
    ]

    return variables


def plot_fig(
    file_name,
    file_num,
    data,
    cl,
    xsize,
    ysize,
    xaxis,
    yaxis,
    xscale,
    yscale,
    xname,
    yname,
    logy,
    var,
    unit,
):

    df = pd.DataFrame(data.T, columns=cl)

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    df[yaxis[1]] = abs(df[yaxis[1]])

    df.plot(x=xaxis, y=yaxis[0], logy=logy[0], ax=ax1, figsize=(xsize, ysize))
    df.plot(x=xaxis, y=yaxis[1], logy=logy[1], ax=ax2, figsize=(xsize, ysize))
    df.plot(x=xaxis, y=yaxis[2], logy=logy[2], ax=ax3, figsize=(xsize, ysize))

    ax1.set_xlabel(xname)
    ax1.set_ylabel(yname[0])
    ax2.set_xlabel(xname)
    ax2.set_ylabel(yname[1])
    ax3.set_xlabel(xname)
    ax3.set_ylabel(yname[2])
    ax1.set_xlim(xscale[0], xscale[1])
    ax2.set_xlim(xscale[0], xscale[1])
    ax3.set_xlim(xscale[0], xscale[1])
    ax1.set_ylim(yscale[0], yscale[1])
    ax2.set_ylim(yscale[2], yscale[3])
    ax3.set_ylim(yscale[4], yscale[5])

    ax1.legend(
        bbox_to_anchor=(0.85, 1.11), loc="upper left", borderaxespad=0, fontsize=18
    )
    ax2.legend(
        bbox_to_anchor=(0.85, 1.11), loc="upper left", borderaxespad=0, fontsize=18
    )
    ax3.legend(
        bbox_to_anchor=(0.85, 1.11), loc="upper left", borderaxespad=0, fontsize=18
    )

    ax1.legend().set_visible(False)
    ax2.legend().set_visible(False)
    ax3.legend().set_visible(False)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "stixsans"
    fig.patch.set_facecolor("white")
    fig.tight_layout()

    #     plt.text(0.01, 0.99, horizontalalignment='left', verticalalignment='top', family='monospace', transform=ax1.transAxes, fontsize=18)

    print "\r", "searching folder...",
    flag = False
    try:
        os.mkdir(save_path + file_name)
    except Exception:
        pass

    print "\r", "saving...",

    try:
        plt.savefig(
            save_path
            + file_name
            + "//"
            + file_name
            + "_"
            + str(file_num)
            + " at "
            + str(var, 4)
            + " "
            + unit
            + ".png"
        )
        flag = True
    except Exception:
        flag = False
        pass


def slack_post(file_path, file_number, date, scan_name, sweep_values):

    print "\r", "slack plotting...",
    try:
        files = {
            "file": open(
                save_path
                + file_name
                + "//"
                + file_name
                + "_"
                + str(file_num)
                + " at "
                + str(var, 4)
                + " "
                + unit
                + ".png",
                "rb",
            )
        }
        param = {
            "token": TOKEN,
            "channels": CHANNEL,
            "title": file_name
            + "_"
            + str(file_num)
            + " at "
            + str(var, 4)
            + " "
            + unit,
        }
        requests.post(
            url="https://slack.com/api/files.upload", params=param, files=files
        )
    except Exception:
        print "\r", "slack plotting failed",
        pass


def sim_sweep(out_ch, vstart, vend, points, delay):
    vs = [[0] * points]
    vs = np.linspace(vstart, vend, num=points)
    d_tmp = None
    p1, p2, p3 = 0, 0, 0

    for jj in range(points):
        sim.dc_set_voltage(out_ch, float("{0:.4f}".format(vs[jj])))
        # time.sleep(delay*0.9)
        try:
            # line_data = [DAC.read_voltage(k) for k in in_dac_ch]
            p1, p2, p3 = LA1.x(), 1.0, 1.0
            line_data = [p1, p2, p3]
        except:
            line_data = [p1, p2, p3]

        if d_tmp is not None:
            d_tmp = np.vstack([d_tmp, line_data])
        else:
            d_tmp = line_data

    return d_tmp.T


def sim_dual_sweep(
    out_ch_bottom,
    out_ch_top,
    vbg_start,
    vbg_end,
    vtg_start,
    vtg_end,
    points_vbg,
    points_vtg,
    delay,
):
    vbg_s = np.linspace(vbg_start, vbg_end, num=points_vbg)
    vtg_s = np.linspace(vtg_start, vtg_end, num=points_vtg)
    d_tmp = None
    p1, p2, p3 = 0, 0, 0

    for jj in range(points_vbg):
        sim.dc_set_voltage(out_ch_bottom, float("{0:.4f}".format(vbg_s[jj])))
        sim.dc_set_voltage(out_ch_top, float("{0:.4f}".format(vtg_s[jj])))
        time.sleep(delay * 0.7)
        try:
            # line_data = [DAC.read_voltage(k) for k in in_dac_ch]
            p1, p2, p3 = LA1.x(), LA2.x(), 1.0
            line_data = [p1, p2, p3]
        except:
            line_data = [p1, p2, p3]

        if d_tmp is not None:
            d_tmp = np.vstack([d_tmp, line_data])
        else:
            d_tmp = line_data

    return d_tmp.T


def set_Vg_nodac(voltage_source, voltage_channel, start_v, end_v):

    if voltage_source == "DAC":
        DAC.buffer_ramp(
            [voltage_channel], [4, 5, 6, 7], [start_v], [end_v], 100, 500, 1
        )
        time.sleep(1)
        print "\r", "Voltage reached: ", end_v, " V",

    elif voltage_source == "SIM":
        sim_sweep(voltage_channel, start_v, end_v, 100, 0.02)
        time.sleep(1)
        print "\r", "Voltage reached: ", end_v, " V",


def set_Vg_dac(voltage_source, voltage_channel, start_v, end_v):

    if voltage_source == "DAC":
        DAC.buffer_ramp(
            [voltage_channel],
            [adc_channel_0, adc_channel_1, adc_channel_2, adc_channel_3],
            [start_v],
            [end_v],
            100,
            500,
            1,
        )
        time.sleep(1)
        print "\r", "Voltage reached: ", end_v, " V",

    elif voltage_source == "SIM":
        sim_sweep2(
            voltage_channel,
            [adc_channel_0, adc_channel_1, adc_channel_2, adc_channel_3],
            start_v,
            end_v,
            100,
            500 / 1e6,
        )
        time.sleep(1)
        print "\r", "Voltage reached: ", end_v, " V",


def scan_Vg(
    voltage_source,
    meas_voltage_gain,
    voltage_channel,
    start_v,
    end_v,
    number_of_vg_points,
    wait_time,
):

    vg = np.linspace(start_v, end_v, number_of_vg_points)

    print "\r", "Scanning Vg:  Start_V:", start_v, " V; End_V:", end_v, " V",
    if voltage_source == "DAC":
        res = DAC.buffer_ramp(
            [voltage_channel],
            [adc_channel_0, adc_channel_1, adc_channel_2, adc_channel_3],
            [start_v],
            [end_v],
            number_of_vg_points,
            wait_time,
            1,
        )

        aux_1, aux_2, aux_3, _ = res

        res_1 = aux_1 * sensitivity_1 / 10.0
        res_2 = aux_2 * sensitivity_2 / 10.0 / meas_voltage_gain
        res_3 = aux_3 * sensitivity_3 / 10.0 / meas_voltage_gain

        # Calculate resistance
        res_5 = np.float64(1.0) * res_2 / res_1
        res_6 = np.float64(1.0) * res_3 / res_1

        # Calculate conductance
        res_7 = np.float64(1.0) / res_5
        res_8 = np.float64(1.0) / res_6

        return np.array([res_1, res_2, res_3, res_5, res_6, res_7, res_8])

    elif voltage_source == "SIM":
        res = sim_sweep(
            voltage_channel, start_v, end_v, number_of_vg_points, wait_time / 1e6
        )

        #         res = sim_sweep2(dc_channel_0, [adc_channel_0, adc_channel_1,adc_channel_2,adc_channel_3], start_v, end_v, number_of_vg_points, wait_time/1e6)

        res_1, res_2, res_3 = res

        # Calculate resistance
        res_5 = np.float64(1.0) * res_2 / res_1 / meas_voltage_gain
        res_6 = np.float64(1.0) * res_3 / res_1 / meas_voltage_gain

        # Calculate conductance
        res_7 = np.float64(1.0) / res_5
        res_8 = np.float64(1.0) / res_6

        return np.array([res_1, res_2, res_3, res_5, res_6, res_7, res_8])


def scan_Vg_one(
    voltage_source,
    meas_voltage_gain,
    amplitude,
    voltage_channel,
    start_v,
    end_v,
    number_of_vg_points,
    wait_time,
):

    vg = np.linspace(start_v, end_v, number_of_vg_points)

    print "\r", "Scanning Vg:  Start_V:", start_v, " V; End_V:", end_v, " V",
    if voltage_source == "DAC":
        res = DAC.buffer_ramp(
            [voltage_channel],
            [adc_channel_0, adc_channel_1, adc_channel_2, adc_channel_3],
            [start_v],
            [end_v],
            number_of_vg_points,
            wait_time,
            1,
        )

        aux_1, aux_2, aux_3, _ = res

        res_1 = aux_1 * sensitivity_1 / 10.0
        res_2 = aux_2 * sensitivity_2 / 10.0 / meas_voltage_gain
        res_3 = aux_3 * sensitivity_3 / 10.0 / meas_voltage_gain

        # Calculate resistance
        res_5 = np.float64(1.0) * res_2 / res_1
        res_6 = np.float64(1.0) * res_3 / res_1

        # Calculate conductance
        res_7 = np.float64(1.0) / res_5
        res_8 = np.float64(1.0) / res_6

        return np.array([res_1, res_2, res_3, res_5, res_6, res_7, res_8])

    elif voltage_source == "SIM":
        res = sim_sweep(
            voltage_channel, start_v, end_v, number_of_vg_points, wait_time / 1e6
        )

        #         res = sim_sweep2(dc_channel_0, [adc_channel_0, adc_channel_1,adc_channel_2,adc_channel_3], start_v, end_v, number_of_vg_points, wait_time/1e6)

        res_1, res_2, res_3 = res

        current = amplitude / 1.0e8

        # Calculate resistance
        res_5 = np.float64(1.0) * res_1 / current / meas_voltage_gain
        res_6 = np.float64(1.0) * res_3 / res_1 / meas_voltage_gain

        # Calculate conductance
        res_7 = np.float64(1.0) / res_5
        res_8 = np.float64(1.0) / res_6

        return np.array([res_1, res_2, res_3, res_5, res_6, res_7, res_8])


def scan_Vg_dual(
    voltage_source,
    meas_voltage_gain,
    out_ch_bottom,
    out_ch_top,
    vbg_start,
    vbg_end,
    vtg_start,
    vtg_end,
    points_vbg,
    points_vtg,
    wait_time,
):

    if voltage_source == "DAC":
        res = DAC.buffer_ramp(
            [voltage_channel],
            [adc_channel_0, adc_channel_1, adc_channel_2, adc_channel_3],
            [start_v],
            [end_v],
            number_of_vg_points,
            wait_time,
            1,
        )

        aux_1, aux_2, aux_3, _ = res

        res_1 = aux_1 * sensitivity_1 / 10.0
        res_2 = aux_2 * sensitivity_2 / 10.0 / meas_voltage_gain
        res_3 = aux_3 * sensitivity_3 / 10.0 / meas_voltage_gain

        # Calculate resistance
        res_5 = np.float64(1.0) * res_2 / res_1
        res_6 = np.float64(1.0) * res_3 / res_1

        # Calculate conductance
        res_7 = np.float64(1.0) / res_5
        res_8 = np.float64(1.0) / res_6

        return np.array([res_1, res_2, res_3, res_5, res_6, res_7, res_8])

    elif voltage_source == "SIM":
        res = sim_dual_sweep(
            out_ch_bottom,
            out_ch_top,
            vbg_start,
            vbg_end,
            vtg_start,
            vtg_end,
            points_vbg,
            points_vtg,
            wait_time / 1e6,
        )

        #         res = sim_sweep2(dc_channel_0, [adc_channel_0, adc_channel_1,adc_channel_2,adc_channel_3], start_v, end_v, number_of_vg_points, wait_time/1e6)

        res_1, res_2, res_3 = res

        # Calculate resistance
        res_5 = np.float64(1.0) * res_2 / res_1 / meas_voltage_gain
        res_6 = np.float64(1.0) * res_3 / res_1 / meas_voltage_gain

        # Calculate conductance
        res_7 = np.float64(1.0) / res_5
        res_8 = np.float64(1.0) / res_6

        return np.array([res_1, res_2, res_3, res_5, res_6, res_7, res_8])


def scan_Vg_dual_one(
    voltage_source,
    meas_voltage_gain,
    amplitude,
    out_ch_bottom,
    out_ch_top,
    vbg_start,
    vbg_end,
    vtg_start,
    vtg_end,
    points_vbg,
    points_vtg,
    wait_time,
):

    if voltage_source == "DAC":
        res = DAC.buffer_ramp(
            [voltage_channel],
            [adc_channel_0, adc_channel_1, adc_channel_2, adc_channel_3],
            [start_v],
            [end_v],
            number_of_vg_points,
            wait_time,
            1,
        )

        aux_1, aux_2, aux_3, _ = res

        res_2 = aux_2 * sensitivity_2 / 10.0 / meas_voltage_gain
        res_3 = aux_3 * sensitivity_3 / 10.0 / meas_voltage_gain

        # Calculate resistance
        res_5 = np.float64(1.0) * res_2 / res_1
        res_6 = np.float64(1.0) * res_3 / res_1

        # Calculate conductance
        res_7 = np.float64(1.0) / res_5
        res_8 = np.float64(1.0) / res_6

        return np.array([res_1, res_2, res_3, res_5, res_6, res_7, res_8])

    elif voltage_source == "SIM":
        res = sim_dual_sweep(
            out_ch_bottom,
            out_ch_top,
            vbg_start,
            vbg_end,
            vtg_start,
            vtg_end,
            points_vbg,
            points_vtg,
            wait_time / 1e6,
        )

        #         res = sim_sweep2(dc_channel_0, [adc_channel_0, adc_channel_1,adc_channel_2,adc_channel_3], start_v, end_v, number_of_vg_points, wait_time/1e6)

        res_1, res_2, res_3 = res

        current = amplitude / 1.0e8

        # Calculate resistance
        res_5 = np.float64(1.0) * res_1 / current / meas_voltage_gain
        res_6 = np.float64(1.0) * res_2 / current / meas_voltage_gain

        # Calculate conductance
        res_7 = np.float64(1.0) / res_5
        res_8 = np.float64(1.0) / res_6

        return np.array([res_1, res_2, res_3, res_5, res_6, res_7, res_8])


def set_Bz(MGz, target_bz):
    # MGz.conf_ramp_rate_field(1, 0.0406, 9.0)
    MGz.conf_field_targ(float(target_bz))
    MGz.ramp()
    print "\r", "Ramping magnetic field to ", target_bz, " T",

    flag = True
    while flag:
        try:
            actual_field = float(MGz.get_field_mag())
            flag = False
        except:
            flag = True

    while abs(float(target_bz) - float(MGz.get_field_mag())) > 1.0e-4:
        continue

    time.sleep(1)
    print "\r", "Magnetic field reached: ", target_bz, " T",
    return float(MGz.get_field_mag())


def set_BxByBz(MGx, MGy, MGz, target_bx, target_by, target_bz):
    # MGz.conf_ramp_rate_field(1, 0.0606, 9.0)
    # MG.conf_ramp_rate_field(1, 0.0806, 12.0)
    time.sleep(0.2)
    MGz.conf_field_targ(float(target_bz))
    MGz.ramp()
    time.sleep(0.2)
    MGy.conf_field_targ(float(target_by))
    MGy.ramp()
    time.sleep(0.2)
    MGx.conf_field_targ(float(target_bx))
    MGx.ramp()
    time.sleep(0.2)
    print "\r", "Ramping magnetic field (Bx, By, Bz): ", target_bx, target_by, target_bz, " T",

    flag = True
    while flag:
        try:
            actual_fieldz = float(MGz.get_field_mag())
            time.sleep(0.2)
            actual_fieldy = float(MGy.get_field_mag())
            time.sleep(0.2)
            actual_fieldx = float(MGx.get_field_mag())
            time.sleep(0.2)
            flag = False
        except:
            flag = True

    while (
        abs(float(target_bz) - actual_fieldz) > 1.0e-4
        or abs(float(target_by) - actual_fieldy) > 1.0e-4
        or abs(float(target_bx) - actual_fieldx) > 1.0e-4
    ):
        actual_fieldz = float(MGz.get_field_mag())
        time.sleep(0.2)
        actual_fieldy = float(MGy.get_field_mag())
        time.sleep(0.2)
        actual_fieldx = float(MGx.get_field_mag())
        time.sleep(0.2)
        print "\r", "current magnetic field (Bx, By, Bz): ", actual_fieldx, actual_fieldy, actual_fieldz, " T", "target magnetic field: (Bx, By, Bz): ", target_bx, target_by, target_bz, " T",

    time.sleep(0.5)
    print "\r", "Magnetic field reached: (Bx, By, Bz): ", target_bx, target_by, target_bz, " T",
    return (
        float(MGx.get_field_mag()),
        float(MGy.get_field_mag()),
        float(MGz.get_field_mag()),
    )


def sweep_Bz(b_start, b_end, points):
    bs = np.linspace(b_start, b_end, num=points)
    d_tmp = None
    p1, p2, p3 = 0, 0, 0

    set_Bz(MGz, b_start)
    # time.sleep(180)
    for i in range(points):
        set_Bz(MGz, bs[i])
        # time.sleep(delay*0.9)
        try:
            # line_data = [DAC.read_voltage(k) for k in in_dac_ch]
            p1, p2, p3 = LA1.x(), 1.0, 1.0
            line_data = [p1, p2, p3]
        except:
            line_data = [p1, p2, p3]

        if d_tmp is not None:
            d_tmp = np.vstack([d_tmp, line_data])
        else:
            d_tmp = line_data

    return d_tmp.T


def scan_B_dual_one(meas_voltage_gain, amplitude, b_start, b_end, points_b):

    res = sweep_Bz(b_start=b_start, b_end=b_end, points=points_b)

    res_1, res_2, res_3 = res

    current = amplitude / 1.0e8

    # Calculate resistance
    res_5 = np.float64(1.0) * res_1 / current / meas_voltage_gain
    res_6 = np.float64(1.0) * res_3 / res_1 / meas_voltage_gain

    # Calculate conductance
    res_7 = np.float64(1.0) / res_5
    res_8 = np.float64(1.0) / res_6

    return np.array([res_1, res_2, res_3, res_5, res_6, res_7, res_8])


def read_T():
    reconnected = False
    while not reconnected:
        try:
            time.sleep(1.0)
            print "\r", "Connecting to temperature server ..."
            # cxn4 = labrad.connect('evaporator-PC', 7682, password='pass')
            cxn4 = labrad.connect()
            tc = cxn4.lakeshore_372
            tc.select_device()
            Tmc, T_p = tc.mc(), tc.probe()
            print "\r", "MXC: ", Tmc, "probe: ", T_p
            time.sleep(1.0)
            print "\r", "Reconnected successfully",
            return Tmc, T_p
        except Exception as e:
            print "\r", str(e),
            print "\r", "Could not reconnect to temperature server",
            time.sleep(2.0)


def set_T(setpoint):
    setpoint_updated = False
    while not setpoint_updated:
        try:
            expression = "SETP 0, %03.2E\n" % setpoint
            # print '\r', expression
            tc.write(expression)  # set the power for the current zone
            Tmc, T_p = tc.mc(), tc.probe()
            print "\r", "MXC: ", Tmc, "probe: ", T_p,
            if setpoint < 1.0:
                temperature_error = min(setpoint * 0.03, 0.01)
            elif setpoint > 6.5 and setpoint < 10.0:
                temperature_error = 0.3
            elif setpoint > 10.0:
                temperature_error = 1.0
            else:
                temperature_error = 0.1

            while abs(Tmc - setpoint) > temperature_error:
                time.sleep(2.0)
                Tmc, T_p = tc.mc(), tc.probe()
                print "\r", "current MXC: ", Tmc, "current probe: ", T_p,

            print "\r", "Target temperature reached.",
            return Tmc, T_p
            setpoint_updated = True

        except Exception as e:
            print "\r", str(e),
            print "\r", "Failed to update setpoint",
            reconnected = False
            while not reconnected:
                try:
                    print "\r", "Connecting to temperature server ...",
                    # cxn4 = labrad.connect('evaporator-PC', 7682, password='pass')
                    cxn4 = labrad.connect()
                    tc = cxn4.lakeshore_372
                    tc.select_device()
                    Tmc, T_p = tc.mc(), tc.probe()
                    print "\r", "MXC: ", Tmc, "probe: ", T_p,
                    print "\r", "Reconnected successfully",
                    reconnected = True
                except Exception as e:
                    print "\r", str(e),
                    print "\r", "Could not reconnect to temperature server",
                    time.sleep(2.0)


def scan_R_vs_Vg_Bz(
    file_path,
    voltage_source,
    voltage_channel,
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
):

    # Get date, parameters and scan name
    cxn0 = labrad.connect()
    DV = cxn0.data_vault
    date1 = datetime.datetime.now()
    meas_parameters = get_meas_parameters()
    scan_name = sys._getframe().f_code.co_name

    # Initial settings of lockins
    set_lockin_parameters(amplitude, frequency)

    # Create data file and save measurement parameters
    scan_var = ("Vg_ind", "Vg", "Bz_ind", "Bz", "Bz_ac", "Tmc", "Tp")
    meas_var = ("Ix", "V1", "V2", "R1", "R2", "G1", "G2")
    file_number = create_file(DV, file_path, scan_name, scan_var, meas_var)
    write_meas_parameters(
        DV,
        file_path,
        file_number,
        date1,
        scan_name,
        meas_parameters,
        amplitude,
        frequency,
    )

    # Create meshes
    # t_lines = np.linspace(t_range[0], t_range[1], number_of_t_lines)
    b_lines = np.linspace(bz_range[0], bz_range[1], number_of_bz_lines)

    t_mc0, t_p0 = 0, 0
    ##### Measurements start #####
    # go to initial gate volatge
    set_Vg_nodac(voltage_source, voltage_channel, 0.0, vg_range[0])

    for ind, val in enumerate(b_lines, 1):
        actual_B = set_Bz(MGz, val)
        print "\r", "Field Line:", ind, "out of ", number_of_bz_lines

        vg_ind = np.linspace(1, number_of_vg_points, number_of_vg_points)
        vg = gate_gain * np.linspace(vg_range[0], vg_range[1], number_of_vg_points)
        b_ind = np.linspace(ind, ind, number_of_vg_points)
        b_val = val * np.ones(number_of_vg_points)
        b_ac = actual_B * np.ones(number_of_vg_points)
        t_mc = t_mc0 * np.ones(number_of_vg_points)
        t_p = t_p0 * np.ones(number_of_vg_points)

        data1 = np.array([vg_ind, vg, b_ind, b_val, b_ac, t_mc, t_p])
        # Scan Vg and acquire data
        data2 = scan_Vg(
            voltage_source,
            meas_voltage_gain,
            voltage_channel,
            vg_range[0],
            vg_range[1],
            number_of_vg_points,
            wait_time,
        )

        data = np.vstack((data1, data2))
        DV.add(data.T)

        plot_fig(
            file_name=scan_name,
            file_num=file_number,
            data=data,
            cl=list(scan_var) + list(meas_var),
            xsize=12,
            ysize=16,
            xaxis="Vg",
            yaxis=["Ix", "R1", "R2"],
            xscale=[None, None],
            yscale=[None, None, 0, 1000, 0, 1000],
            xname="Vg",
            yname=["Ix", "R1", "R2"],
            logy=[False, False, False],
            var=0,
            unit="T",
        )
        # go to next gate voltage
        if ind < number_of_bz_lines:
            set_Vg_nodac(voltage_source, voltage_channel, vg_range[1], vg_range[0])

    # go to 0 V
    set_Vg_nodac(voltage_source, voltage_channel, vg_range[1], 0.0)
    print "\r", "measurement number: ", file_number, scan_name, " done"
    ##### Measurements done #####
    date2 = datetime.datetime.now()
    write_meas_parameters_end(date1, date2, file_path)


def scan_R_vs_n_Bz_at_fixedD(
    file_path,
    voltage_source,
    voltage_channel_bottom,
    voltage_channel_top,
    amplitude=0.01,
    frequency=17.777,
    gate_gain=1.0,
    meas_voltage_gain=1.0,
    displacement_field=0.0,
    bz_range=[0.0, 0.0],
    n_range=[-1.0, 1.0],
    number_of_bz_lines=1,
    number_of_n_points=200,
    wait_time=wait_time,
    c_delta=0.0,
    note="misc",
):

    # Get date, parameters and scan name
    cxn0 = labrad.connect()
    DV = cxn0.data_vault
    date1 = datetime.datetime.now()
    meas_parameters = get_meas_parameters()
    scan_name = sys._getframe().f_code.co_name

    # Initial settings of lockins
    set_lockin_parameters(amplitude, frequency)

    # Create data file and save measurement parameters
    scan_var = ("n_ind", "n", "Bz_ind", "iBz", "Bz", "Tmc", "Tp")
    meas_var = ("Ix", "V1", "V2", "R1", "R2", "G1", "G2")
    file_number = create_file(DV, file_path, scan_name, scan_var, meas_var)
    write_meas_parameters(
        DV,
        file_path,
        file_number,
        date1,
        scan_name,
        meas_parameters,
        amplitude,
        frequency,
    )

    # Create meshes
    # t_lines = np.linspace(t_range[0], t_range[1], number_of_t_lines)
    b_lines = np.linspace(bz_range[0], bz_range[1], number_of_bz_lines)

    t_mc0, t_p0 = 0, 0
    D_val = displacement_field
    ##### Measurements start #####
    # go to initial gate volatge
    vtg_last, vbg_last = 0, 0
    for ind, val in enumerate(b_lines, 1):
        d3 = datetime.datetime.now()
        actual_B = set_Bz(MGz, val)
        print "\r", "Field Line:", ind, "out of ", number_of_bz_lines
        vtg_s, vbg_s = find_vt_vb(D_val, n_range[0], c_delta)
        vtg_e, vbg_e = find_vt_vb(D_val, n_range[1], c_delta)
        sim_dual_sweep(
            out_ch_bottom=voltage_channel_bottom,
            out_ch_top=voltage_channel_top,
            vbg_start=vbg_last,
            vbg_end=vbg_s,
            vtg_start=vtg_last,
            vtg_end=vtg_s,
            points_vbg=40,
            points_vtg=40,
            delay=0.005,
        )

        n_ind = np.linspace(1, number_of_n_points, number_of_n_points)
        n = gate_gain * np.linspace(n_range[0], n_range[1], number_of_n_points)
        b_ind = np.linspace(ind, ind, number_of_n_points)
        b_val = val * np.ones(number_of_n_points)
        b_ac = actual_B * np.ones(number_of_n_points)
        ib_val = np.float64(1.0 / actual_B) * np.ones(number_of_n_points)
        t_mc = t_mc0 * np.ones(number_of_n_points)
        t_p = t_p0 * np.ones(number_of_n_points)

        data1 = np.array([n_ind, n, b_ind, ib_val, b_val, t_mc, t_p])
        # Scan Vg and acquire data

        data2 = scan_Vg_dual_one(
            voltage_source,
            meas_voltage_gain,
            amplitude=amplitude,
            out_ch_bottom=voltage_channel_bottom,
            out_ch_top=voltage_channel_top,
            vbg_start=vbg_s,
            vbg_end=vbg_e,
            vtg_start=vtg_s,
            vtg_end=vtg_e,
            points_vbg=number_of_n_points,
            points_vtg=number_of_n_points,
            wait_time=wait_time,
        )
        t = datetime.datetime.now() - d3
        print "\r", "one epoch time:", t, "estimated finish time:", datetime.datetime.now() + (
            len(b_lines) - ind
        ) * t
        vtg_last, vbg_last = vtg_e, vbg_e
        data = np.vstack((data1, data2))
        DV.add(data.T)

        plot_fig(
            file_name=scan_name,
            file_num=file_number,
            data=data,
            cl=list(scan_var) + list(meas_var),
            xsize=12,
            ysize=16,
            xaxis="n",
            yaxis=["Ix", "R1", "R2"],
            xscale=[None, None],
            yscale=[None, None, 10000, 100000, 0, 1000],
            xname="n",
            yname=["Ix", "R1", "R2"],
            logy=[False, False, False],
            var=0,
            unit="T",
        )

        # go to next gate voltage

    # go to 0 V
    sim_dual_sweep(
        out_ch_bottom=voltage_channel_bottom,
        out_ch_top=voltage_channel_top,
        vbg_start=vbg_last,
        vbg_end=0.0,
        vtg_start=vtg_last,
        vtg_end=0.0,
        points_vbg=40,
        points_vtg=40,
        delay=0.005,
    )
    print "\r", "measurement number: ", file_number, scan_name, " done"
    ##### Measurements done #####
    date2 = datetime.datetime.now()
    write_meas_parameters_end(date1, date2, file_path)


def scan_R_vs_n_D_fixedBz(
    file_path,
    voltage_source,
    voltage_channel_bottom,
    voltage_channel_top,
    amplitude=0.01,
    frequency=17.777,
    gate_gain=1.0,
    meas_voltage_gain=1.0,
    magnetic_field=0.0,
    n_range=[-0.5, 0.5],
    D_range=[-1.0, 1.0],
    number_of_D_lines=1,
    number_of_n_points=200,
    wait_time=wait_time,
    c_delta=0.0,
    note="misc",
):
    # Get date, parameters and scan name
    cxn0 = labrad.connect()
    DV = cxn0.data_vault
    date1 = datetime.datetime.now()
    meas_parameters = get_meas_parameters()
    scan_name = sys._getframe().f_code.co_name

    # Initial settings of lockins
    set_lockin_parameters(amplitude, frequency)

    # Create data file and save measurement parameters
    scan_var = ("n_ind", "n", "D_ind", "D", "Tmc", "Tp")
    meas_var = ("Ix", "V1", "V2", "R1", "R2", "G1", "G2")
    file_number = create_file(DV, file_path, scan_name, scan_var, meas_var)
    write_meas_parameters(
        DV,
        file_path,
        file_number,
        date1,
        scan_name,
        meas_parameters,
        amplitude,
        frequency,
    )
    DV.add_parameter("live_plots", [("n", "D", "R1")])

    # Create meshes
    # t_lines = np.linspace(t_range[0], t_range[1], number_of_t_lines)
    D_lines = np.linspace(D_range[0], D_range[1], number_of_D_lines)

    t_mc0, t_p0 = 0, 0
    ##### Measurements start #####
    set_Bz(MGz, magnetic_field)
    vtg_last, vbg_last = 0, 0
    for ind, D_val in enumerate(D_lines):
        d3 = datetime.datetime.now()
        vtg_s, vbg_s = find_vt_vb(D_val, n_range[0], c_delta)
        vtg_e, vbg_e = find_vt_vb(D_val, n_range[1], c_delta)
        sim_dual_sweep(
            out_ch_bottom=voltage_channel_bottom,
            out_ch_top=voltage_channel_top,
            vbg_start=vbg_last,
            vbg_end=vbg_s,
            vtg_start=vtg_last,
            vtg_end=vtg_s,
            points_vbg=40,
            points_vtg=40,
            delay=0.005,
        )
        #         if ind == 0: set_Vg_nodac(voltage_source, voltage_channel_top, 0.0, vt_start)
        #         else: set_Vg_nodac(voltage_source, voltage_channel_top, vgt_lines[ind-1], val)
        print "\r", "D Line:", ind + 1, "out of ", number_of_D_lines

        n_ind = np.linspace(1, number_of_n_points, number_of_n_points)
        n = gate_gain * np.linspace(n_range[0], n_range[1], number_of_n_points)
        D_ind = np.linspace(ind, ind, number_of_n_points)
        D = D_val * np.ones(number_of_n_points)
        t_mc = t_mc0 * np.ones(number_of_n_points)
        t_p = t_p0 * np.ones(number_of_n_points)

        data1 = np.array([n_ind, n, D_ind, D, t_mc, t_p])
        # Scan Vg and acquire data

        data2 = scan_Vg_dual_one(
            voltage_source,
            meas_voltage_gain,
            amplitude=amplitude,
            out_ch_bottom=voltage_channel_bottom,
            out_ch_top=voltage_channel_top,
            vbg_start=vbg_s,
            vbg_end=vbg_e,
            vtg_start=vtg_s,
            vtg_end=vtg_e,
            points_vbg=number_of_n_points,
            points_vtg=number_of_n_points,
            wait_time=wait_time,
        )
        vtg_last, vbg_last = vtg_e, vbg_e

        t = datetime.datetime.now() - d3
        print "\r", "one epoch time:", t, "estimated finish time:", datetime.datetime.now() + (
            len(D_lines) - ind
        ) * t

        data = np.vstack((data1, data2))
        DV.add(data.T)

    sim_dual_sweep(
        out_ch_bottom=voltage_channel_bottom,
        out_ch_top=voltage_channel_top,
        vbg_start=vbg_last,
        vbg_end=0.0,
        vtg_start=vtg_last,
        vtg_end=0.0,
        points_vbg=40,
        points_vtg=40,
        delay=0.005,
    )
    set_Bz(MGz, 0.0)
    print "\r", "measurement number: ", file_number, scan_name, " done"
    ##### Measurements done #####
    date2 = datetime.datetime.now()
    write_meas_parameters_end(date1, date2, file_path)


def scan_R_vs_B_fixed_n_D(
    file_path,
    voltage_source,
    voltage_channel_bottom,
    voltage_channel_top,
    amplitude=0.01,
    frequency=17.777,
    gate_gain=1.0,
    meas_voltage_gain=1.0,
    n_range=[0, 0],
    D_range=[0, 0],
    b_range=[-0.1, 0.1],
    number_of_n_points=1,
    number_of_D_points=1,
    number_of_b_points=100,
    wait_time=wait_time,
    c_delta=0.0,
    note="misc",
):

    # MGz.conf_ramp_rate_field(0.018060)
    # Get date, parameters and scan name
    cxn0 = labrad.connect()
    DV = cxn0.data_vault
    date1 = datetime.datetime.now()
    meas_parameters = get_meas_parameters()
    scan_name = sys._getframe().f_code.co_name

    # Initial settings of lockins
    set_lockin_parameters(amplitude, frequency)

    # Create data file and save measurement parameters
    scan_var = ("B_ind", "B", "n_ind", "D_ind", "n", "D", "Tmc", "Tp")
    meas_var = ("Ix", "V1", "V2", "R1", "R2", "G1", "G2")

    n_lines = np.linspace(n_range[0], n_range[1], number_of_n_points)
    D_lines = np.linspace(D_range[0], D_range[1], number_of_D_points)

    t_mc0, t_p0 = 0, 0
    ##### Measurements start #####
    # go to initial gate volatge
    vtg_last, vbg_last = 0, 0
    n_mesh, D_mesh = np.meshgrid(n_lines, D_lines, sparse=False, indexing="ij")
    for i in range(number_of_n_points):
        for j in range(number_of_D_points):
            n, D = n_mesh[i, j], D_mesh[i, j]
            vtg_s, vbg_s = find_vt_vb(D, n, c_delta)

            sim_dual_sweep(
                out_ch_bottom=voltage_channel_bottom,
                out_ch_top=voltage_channel_top,
                vbg_start=vbg_last,
                vbg_end=vbg_s,
                vtg_start=vtg_last,
                vtg_end=vtg_s,
                points_vbg=40,
                points_vtg=40,
                delay=0.005,
            )

            b_ind = np.linspace(1, number_of_b_points, number_of_b_points)
            b = np.linspace(b_range[0], b_range[1], number_of_b_points)
            br = b[::-1]
            n_ind = i * np.ones(number_of_b_points)
            D_ind = j * np.ones(number_of_b_points)
            n_val = n * np.ones(number_of_b_points)
            D_val = D * np.ones(number_of_b_points)
            t_mc = t_mc0 * np.ones(number_of_b_points)
            t_p = t_p0 * np.ones(number_of_b_points)

            file_number1 = create_file(
                DV, file_path, scan_name + "_trace", scan_var, meas_var
            )
            write_meas_parameters(
                DV,
                file_path,
                file_number1,
                date1,
                scan_name + "_trace",
                meas_parameters,
                amplitude,
                frequency,
            )

            data1 = np.array([b_ind, b, n_ind, D_ind, n_val, D_val, t_mc, t_p])
            data2 = scan_B_dual_one(
                meas_voltage_gain=meas_voltage_gain,
                amplitude=amplitude,
                b_start=b_range[0],
                b_end=b_range[1],
                points_b=number_of_b_points,
            )
            data = np.vstack((data1, data2))
            DV.add(data.T)

            file_number2 = create_file(
                DV, file_path, scan_name + "_retrace", scan_var, meas_var
            )
            write_meas_parameters(
                DV,
                file_path,
                file_number2,
                date1,
                scan_name + "_retrace",
                meas_parameters,
                amplitude,
                frequency,
            )

            data1 = np.array([b_ind, br, n_ind, D_ind, n_val, D_val, t_mc, t_p])
            data2 = scan_B_dual_one(
                meas_voltage_gain=meas_voltage_gain,
                amplitude=amplitude,
                b_start=b_range[1],
                b_end=b_range[0],
                points_b=number_of_b_points,
            )
            data = np.vstack((data1, data2))
            DV.add(data.T)

            vtg_last, vbg_last = vtg_s, vbg_s

    sim_dual_sweep(
        out_ch_bottom=voltage_channel_bottom,
        out_ch_top=voltage_channel_top,
        vbg_start=vbg_last,
        vbg_end=0.0,
        vtg_start=vtg_last,
        vtg_end=0.0,
        points_vbg=40,
        points_vtg=40,
        delay=0.005,
    )

    print "\r", "measurement number: ", file_number2, scan_name, " done"
    ##### Measurements done #####
    # MGz.conf_ramp_rate_field(0.05060)

    date2 = datetime.datetime.now()
    write_meas_parameters_end(date1, date2, file_path)
