# -*- coding: utf-8 -*-
import os
import time
import multiprocessing as mp
import numpy as np
import pandas as pd
from project.lhs_max_st_temp.func import mc_fr_calculation, mc_inputs_generator, mc_post_processing
from project.func.temperature_fires import standard_fire_iso834 as standard_fire
from project.cls.plot import Scatter2D
from pickle import load as pload
from pickle import dump as pdump
from project.func.files import list_all_files_with_suffix


# wrapper to deal with inputs format (dict-> kwargs)
def worker(arg):
    kwargs, q = arg
    result = mc_fr_calculation(**kwargs)
    q.put(kwargs)
    return result


def step0_parse_input_files(dir_work):

    list_files = list_all_files_with_suffix(dir_work, ".txt", is_full_dir=False)

    list_input_files = []
    for f_ in list_files:
        with open("/".join([dir_work, f_]), "r") as f__:
            l = f__.readline()
        if l.find("# MC INPUT FILE") > -1:
            list_input_files.append(f_)

    return list_input_files


def step1_inputs_maker(dir_work, file_name, n_sim, T_s_fix=273.15+620):
    key_ = file_name.split(".")[0]
    dir_input_file = "/".join([dir_work, file_name])
    dir_kwargs_file = "{}/{} - {}".format(dir_work, key_, "in_main_calc.p")
    fire = standard_fire(np.arange(0, 3*60*60, 1), 273.15+20)
    inputs_extra = {"beam_temperature_goal": T_s_fix,
                    "iso834_time": fire[0],
                    "iso834_temperature": fire[1],}
    list_kwargs = mc_inputs_generator(n_sim, inputs_extra, dir_input_file)

    # todo: Check if file path that kwargs goes in already exits, delete if it already exist.

    # Save list_kwargs as a pickle object.
    pdump(list_kwargs, open(dir_kwargs_file, "wb"))


def step2_main_calc(dir_work, kwargs_file_name, n_proc=0, progress_print_interval=1):
    # Load kwargs
    dir_kwargs_file = "/".join([dir_work, kwargs_file_name])
    list_kwargs = pload(open(dir_kwargs_file, "rb"))

    # Identify the id string
    id_ = kwargs_file_name.split(" - ")[0]

    # Check number of processes are to be used
    n_proc = os.cpu_count() if int(n_proc) < 1 else int(n_proc)

    # SIMULATION
    print("SIMULATION STARTS")
    print(("{}{}\n"*2).format("Number of Threads:", n_proc, "Total Simulations:", len(list_kwargs)))
    time_count_simulation = time.perf_counter()
    m = mp.Manager()
    q = m.Queue()
    p = mp.Pool(n_proc)
    jobs = p.map_async(worker, [(kwargs, q) for kwargs in list_kwargs])
    count_total_simulations = len(list_kwargs)
    while progress_print_interval:
        if jobs.ready():
            break
        else:
            print("SIMULATION COMPLETE {:3.0f} %...".format(q.qsize() * 100 / count_total_simulations))
            time.sleep(progress_print_interval)
    results = jobs.get()
    time_count_simulation = time.perf_counter() - time_count_simulation
    print("SIMULATION COMPLETE IN {:0.3f} SECONDS".format(time_count_simulation, progress_print_interval))

    # POST PROCESS
    print("POST PROCESSING STARTS")
    # format outputs
    results = np.array(results, dtype=float)
    df_outputs = pd.DataFrame({"TIME EQUIVALENCE [min]": results[:, 0]/60.,
                               "SEEK STATUS [bool]": results[:, 1],
                               "WINDOW OPEN FRACTION [%]": results[:, 2],
                               "FIRE LOAD DENSITY [MJ/m2]": results[:, 3],
                               "FIRE SPREAD SPEED [m/s]": results[:, 4],
                               "BEAM POSITION [m]": results[:, 5],
                               "MAX. NEAR FIELD TEMPERATURE [C]": results[:, 6],
                               "FIRE TYPE [0:P., 1:T.]": results[:, 7],
                               "PEAK STEEL TEMPERATURE [C]": results[:, 8]-273.15,
                               "PROTECTION THICKNESS [m]": results[:, 9],
                               "SEEK ITERATIONS": results[:, 10]})
    df_outputs = df_outputs[["TIME EQUIVALENCE [min]", "PEAK STEEL TEMPERATURE [C]", "PROTECTION THICKNESS [m]", "SEEK STATUS [bool]", "WINDOW OPEN FRACTION [%]", "FIRE LOAD DENSITY [MJ/m2]", "FIRE SPREAD SPEED [m/s]", "BEAM POSITION [m]", "MAX. NEAR FIELD TEMPERATURE [C]", "FIRE TYPE [0:P., 1:T.]"]]
    df_outputs.sort_values(by=["TIME EQUIVALENCE [min]"], inplace=True)
    df_outputs.reset_index(drop=True, inplace=True)

    dir_results_file = "{}/{} - {}".format(dir_work, id_, "res_df.p")
    pdump(df_outputs, open(dir_results_file, "wb"))


def step3_results_numerical(dir_work, obj_file_name):

    # Obtain ID string
    id_ = obj_file_name.split(" - ")[0]

    # Obtain full directory of the dataframe obj file
    dir_obj_file = "{}/{} - {}".format(dir_work, id_, "res_df.p")
    dir_csv_file = "{}/{} - {}".format(dir_work, id_, "res_num.csv")

    # Load the dataframe obj file
    df_results = pload(open(dir_obj_file, "rb"))

    # Save the dataframe to csv file
    df_results.to_csv(dir_csv_file, index=True, sep=",")


def step4_results_visulisation(dir_work, obj_file_name, height_building):

    # Obtain ID string
    id_ = obj_file_name.split(" - ")[0]

    # Obtain full directory of the dataframe obj file
    dir_obj_file = "{}/{} - {}".format(dir_work, id_, "res_df.p")

    # Load the dataframe obj file
    df_results = pload(open(dir_obj_file, "rb"))

    # Obtain time equivalence, in minutes, as x-axis values
    x = df_results["TIME EQUIVALENCE [min]"].values * 60.

    # Filter out entries with failed seek status, i.e. peak steel temperature is not within the tolerance.
    # seek_status = df_results["SEEK STATUS [bool]"].values
    # x = x[seek_status == 1]

    # Define horizontal line(s) to plot
    y__ = 1 - 64.8 / height_building ** 2

    x, y, x_, y_, xy_found = mc_post_processing(x, y_find=[y__])

    plt = Scatter2D()
    plt.plot2(x/60, y, "Simulation results")
    # plt.plot2(x_/60, y_, "Interpolated CDF")
    plt.format(**{"figure_size_scale": 0.7, "axis_lim_y1": (0, 1), "axis_lim_x": (0, 120), "legend_is_shown": False, "axis_label_x": "Time Equivalence [min]", "axis_label_y1": "Fractile", "marker_size": 0})

    # plt.update_line_format("Interpolated CDF", **{"line_width": 0.5, "color": "black", "line_style": ":"})
    x_found, y_found = xy_found[0, :]
    plt.axes_primary.axvline(x=x_found/60., color="black", linewidth=1)
    plt.axes_primary.axhline(y=y_found, color="black", linewidth=1)
    plt.axes_primary.text(x=x_found/60+1, y=y_found-0.01, s="({:.0f}, {:.4f})".format(x_found/60, y_found), va="top", ha="left", fontsize=6)
    # plt.update_legend(legend_loc=0)
    plt.save_figure(file_name=" - ".join([id_, "res_plot"]), file_format=".pdf", dir_folder=dir_work)

    print("POST PROCESSING COMPLETE")


if __name__ == "__main__":
    # SETTINGS
    simulations = 100
    steel_temperature_to_fix = 273.15 + 620
    building_height = 60
    project_full_path = "C:/Users/Ian Fu/Dropbox (OFR-UK)/Bicester_team_projects/Live_projects/Symons House/Time Equivalence Analysis/calc"

    # ROUTINES
    list_files = step0_parse_input_files(dir_work=project_full_path)
    # list_files = [list_files[1]]
    ff = "{} - {}"

    for f in list_files:
        print(f)
        id_ = f.split(".")[0]
        step1_inputs_maker(project_full_path, f, simulations)
        step2_main_calc(project_full_path, ff.format(id_, "in_main_calc.p"), 0, 5)
        step3_results_numerical(project_full_path, ff.format(id_, "res_df.p"))
        step4_results_visulisation(project_full_path, ff.format(id_, "res_df.p"), 60)
