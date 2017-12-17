import os
import time
import multiprocessing as mp
import numpy as np
import pandas as pd
from project.SRPA.func_core import mc_inputs_generator, calc_time_equiv_worker
from project.func.temperature_fires import standard_fire_iso834 as standard_fire
from project.cls.plot import Scatter2D
from pickle import load as pload
from pickle import dump as pdump
from project.func.files import list_all_files_with_suffix
from scipy.interpolate import interp1d


def step0_parse_input_files(dir_work):

    list_files = list_all_files_with_suffix(dir_work, ".txt", is_full_dir=True)

    list_input_files = []
    for f_ in list_files:
        with open(f_, "r") as f__:
            l = f__.readline()
        if l.find("# MC INPUT FILE") > -1:
            list_input_files.append(f_)

    return list_input_files


def step1_inputs_maker(path_input_file):

    file_name = os.path.basename(path_input_file)
    dir_work = os.path.dirname(path_input_file)
    id_ = file_name.split(".")[0]
    path_setting_file = os.path.join(dir_work, "{} - {}".format(id_, "prefs.p"))
    path_variable_file = os.path.join(dir_work, "{} - {}".format(id_, "args_main.p"))
    fire = standard_fire(np.arange(0, 3*60*60, 1), 273.15+20)
    inputs_extra = {"iso834_time": fire[0],
                    "iso834_temperature": fire[1],}
    list_kwargs, dict_settings = mc_inputs_generator(dict_extra_variables_to_add=inputs_extra,
                                                     dir_file=path_input_file)

    # Save list_kwargs as a pickle object.
    pdump(list_kwargs, open(path_variable_file, "wb"))
    pdump(dict_settings, open(path_setting_file, "wb"))


def step2_main_calc(path_input_file, progress_print_interval=1):

    # Make prefix, suffix, file and directory strings
    dir_work = os.path.dirname(path_input_file)
    name_kwargs_file = os.path.basename(path_input_file)
    id_ = name_kwargs_file.split(" - ")[0]

    # Load kwargs
    list_kwargs = pload(open(path_input_file, "rb"))

    # Load settings
    path_settings_file = os.path.join(dir_work, "{} - {}".format(id_, "prefs.p"))
    dict_settings = pload(open(path_settings_file, "rb"))
    n_proc = dict_settings["n_proc"]

    # Check number of processes are to be used
    n_proc = os.cpu_count() if int(n_proc) < 1 else int(n_proc)

    # SIMULATION
    print("SIMULATION STARTS")
    print("{}{}".format("Input file:", id_))
    print("{}{}".format("Total Simulations:", len(list_kwargs)))
    print("{}{}".format("Number of Threads:", n_proc))

    time_count_simulation = time.perf_counter()
    m = mp.Manager()
    q = m.Queue()
    p = mp.Pool(n_proc)
    jobs = p.map_async(calc_time_equiv_worker, [(kwargs, q) for kwargs in list_kwargs])
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
                               "PEAK STEEL TEMPERATURE TO GOAL SEEK [C]": results[:, 8]-273.15,
                               "PROTECTION THICKNESS [m]": results[:, 9],
                               "SEEK ITERATIONS [-]": results[:, 10],
                               "PEAK STEEL TEMPERATURE TO FIXED PROTECTION [C]": np.sort(results[:, 11])-273.15})
    df_outputs = df_outputs[["TIME EQUIVALENCE [min]", "PEAK STEEL TEMPERATURE TO GOAL SEEK [C]", "PROTECTION THICKNESS [m]", "SEEK STATUS [bool]", "SEEK ITERATIONS [-]", "WINDOW OPEN FRACTION [%]", "FIRE LOAD DENSITY [MJ/m2]", "FIRE SPREAD SPEED [m/s]", "BEAM POSITION [m]", "MAX. NEAR FIELD TEMPERATURE [C]", "FIRE TYPE [0:P., 1:T.]", "PEAK STEEL TEMPERATURE TO FIXED PROTECTION [C]"]]
    df_outputs.sort_values(by=["TIME EQUIVALENCE [min]"], inplace=True)
    df_outputs.reset_index(drop=True, inplace=True)

    path_results_file = os.path.join(dir_work, "{} - {}".format(id_, "res_df.p"))
    pdump(df_outputs, open(path_results_file, "wb"))


def step3_results_numerical(path_input_file):
    # Make prefix, suffix, file and directory strings
    dir_cwd = os.path.dirname(path_input_file)
    name_obj_file = os.path.basename(path_input_file)
    id_ = name_obj_file.split(" - ")[0]
    path_csv_file = os.path.join(dir_cwd, "{} - {}".format(id_, "res_num.csv"))

    # Load the dataframe obj file
    df_results = pload(open(path_input_file, "rb"))

    # Save the dataframe to csv file
    df_results.to_csv(path_csv_file, index=True, sep=",")


def step4_results_visulisation(path_input_file):

    # Make file and directory names
    dir_work = os.path.dirname(path_input_file)
    obj_file_name = os.path.basename(path_input_file)
    id_ = obj_file_name.split(" - ")[0]  # Obtain ID string

    # Load settings
    path_settings_file = os.path.join(dir_work, "{} - {}".format(id_, "prefs.p"))
    dict_settings = pload(open(path_settings_file, "rb"))
    height_building = dict_settings["building_height"]

    # Load the dataframe obj file
    df_results = pload(open(path_input_file, "rb"))


    # Filter out entries with failed seek status, i.e. peak steel temperature is not within the tolerance.
    # seek_status = df_results["SEEK STATUS [bool]"].values
    # x = x[seek_status == 1]

    # Define horizontal line(s) to plot
    if height_building > 0:
        y__ = 1 - 64.8 / height_building ** 2
    else:
        y__ = 0.5

    # Obtain time equivalence, in minutes, as x-axis values
    x = df_results["TIME EQUIVALENCE [min]"].values * 60.
    y = np.arange(1, len(x) + 1) / len(x)
    f_interp = interp1d(y, x)
    if height_building > 0:
        y_line = 1 - 64.8 / height_building ** 2
        x_line = f_interp(y_line)
    else:
        x_line = y_line = 0

    plt = Scatter2D()
    plt_format = {"figure_size_scale": 0.7,
                  "axis_lim_y1": (0, 1),
                  "axis_lim_x": (0, 120),
                  "legend_is_shown": True,
                  "axis_label_x": "Time Equivalence [min]",
                  "axis_label_y1": "Fractile",
                  "marker_size": 0}
    plt.plot2(x/60, y, "Simulation results")

    plt.format(**plt_format)

    if height_building > 0:
        x_end = plt.axes_primary.get_xlim()[1]
        y_end = plt.axes_primary.get_ylim()[1]

        x_line_ = x_line/60.
        y_line_ = y_line
        plt.plot_vertical_line(x=x_line_)
        plt.axes_primary.text(x=x_line_, y=y_end, s="{:.0f}".format(x_line_), va="bottom", ha="center", fontsize=6)
        plt.plot_horizontal_line(y=y_line_)
        plt.axes_primary.text(x=x_end, y=y_line_, s="{:.4f}".format(y_line_), va="center", ha="left", fontsize=6)

    plt.save_figure(file_name=" - ".join([id_, "res_plot"]), file_format=".png", dir_folder=dir_work)
    print("RESULTS PLOT SAVED: {}".format(" - ".join([id_, "res_plot"])+".png"))


def step5_results_visulisation_all(dir_work):
    # get all file names within 'dir_work' directory
    results_files = list_all_files_with_suffix(dir_work, " - res_df.p", is_full_dir=False)

    # proceed only if there are more than one files
    if len(results_files) == 1:
        return 0

    # ------------------------------------------------------------------------------------------------------------------
    # Plot a Graph for All Data
    # ------------------------------------------------------------------------------------------------------------------

    # instantiate plotting object
    plt = Scatter2D()

    # format parameters for figure
    plt_format = {"figure_size_scale": 0.4,
                  "axis_lim_y1": (0, 1),
                  "axis_lim_x": (0, 120),
                  "legend_is_shown": True,
                  "axis_label_x": "Time Equivalence [min]",
                  "axis_label_y1": "Fractile",
                  "marker_size": 0}

    # format parameters for additional texts which indicate the x_line and y_line values
    # plt_format_text = {"fontsize": 6, "bbox": dict(boxstyle="square", fc="w", ec="b")}

    # container for x_line and y_line
    x_line, y_line = [], []
    height_building = 0

    # iterate through all result files and plot lines accordingly
    for dir_obj_file in results_files:
        # plot this line
        id_ = dir_obj_file.split(" - ")[0]

        # Load settings
        path_settings_file = os.path.join(dir_work, "{} - {}".format(id_, "prefs.p"))
        dict_settings = pload(open(path_settings_file, "rb"))
        height_building = dict_settings["building_height"]

        # load obj from given file path
        path_obj_file = os.path.join(dir_work, dir_obj_file)
        df_results = pload(open(path_obj_file, "rb"))

        # obtain values: x, y, x_line (vertical line) and y_line (horizontal line)
        x = df_results["TIME EQUIVALENCE [min]"].values
        y = np.arange(1, len(x) + 1) / len(x)
        f_interp = interp1d(y, x)
        if height_building == 0:
            y_line_ = 0
        else:
            y_line_ = 1 - 64.8 / height_building ** 2
            x_line_ = f_interp(y_line_)

        # plot line f(x)
        plt.plot2(x, y, id_)
        plt.format(**plt_format)

        # obtain x_line and y_line for later use
        if height_building > 0:
            x_line.append(round(float(x_line_), 0))
            y_line.append(round(float(y_line_), 4))

    plt.format(**plt_format)

    if height_building > 0:
        x_line = set(x_line)
        y_line = set(y_line)
        x_end = plt.axes_primary.get_xlim()[1] + 5
        y_end = plt.axes_primary.get_ylim()[1] + 0.005

        x_line = [max(x_line)]

        for x_line_ in x_line:
            plt.plot_vertical_line(x=x_line_)
            plt.add_text(x=x_line_, y=y_end, s="{:.0f}".format(x_line_), va="bottom", ha="center", fontsize=6)

        for y_line_ in y_line:
            plt.plot_horizontal_line(y=y_line_)
            plt.add_text(x=x_end, y=y_line_, s="{:.4f}".format(y_line_), va="left", ha="center", fontsize=6)

    plt.plot_vertical_line(x=57)
    plt.add_text(x=57, y=1, s="{}".format("Max. 57"), va="bottom", ha="center", fontsize=6)

    plt.save_figure(dir_folder=dir_work, file_name=os.path.basename(dir_work), file_format=".png")


def step6_fire_curves_pick():
    pass
