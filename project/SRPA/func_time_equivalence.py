import os, copy, time
import multiprocessing as mp
import numpy as np
import pandas as pd
from project.SRPA.func_core import mc_inputs_generator, calc_time_equiv_worker, mc_inputs_generator2
from project.func.temperature_fires import standard_fire_iso834 as standard_fire
from project.cls.plot import Scatter2D
from pickle import load as pload
from pickle import dump as pdump
from project.func.files import list_all_files_with_suffix
from scipy.interpolate import interp1d
from project.SRPA.tfm_alt import travelling_fire as _fire_travelling
from project.func.temperature_fires import parametric_eurocode1 as _fire_param

strformat_1_1 = "{:25}{}"
strformat_1_1_1 = "{:25}{:3}{}"

def saveprint(file_name):
    print(strformat_1_1.format("File saved:", file_name))


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
    # list_kwargs, dict_settings = mc_inputs_generator(dict_extra_variables_to_add=inputs_extra,
    #                                                  dir_file=path_input_file)
    df_args, dict_settings = mc_inputs_generator2(dict_extra_variables_to_add=inputs_extra,
                                                 dir_file=path_input_file)

    # Save list_kwargs as a pickle object.
    pdump(df_args, open(path_variable_file, "wb"))
    saveprint(os.path.basename(path_variable_file))

    pdump(dict_settings, open(path_setting_file, "wb"))
    saveprint(os.path.basename(path_setting_file))


def step2_main_calc(path_input_file, progress_print_interval=1):

    # Make prefix, suffix, file and directory strings
    dir_work = os.path.dirname(path_input_file)
    name_kwargs_file = os.path.basename(path_input_file)
    id_ = name_kwargs_file.split(" - ")[0]

    # Load kwargs
    df_input_kwargs = pload(open(path_input_file, "rb"))
    dict_input_kwargs = df_input_kwargs.to_dict(orient="index")
    list_kwargs = []
    for key, val in dict_input_kwargs.items():
        val["index"] = key
        list_kwargs.append(val)
    # list_kwargs = [val for key, val in dict_input_kwargs.items()]

    # Load settings
    path_settings_file = os.path.join(dir_work, "{} - {}".format(id_, "prefs.p"))
    dict_settings = pload(open(path_settings_file, "rb"))
    n_proc = dict_settings["n_proc"]

    # Check number of processes are to be used
    n_proc = os.cpu_count() if int(n_proc) < 1 else int(n_proc)

    # SIMULATION
    print(strformat_1_1.format("Input file:", id_))
    print(strformat_1_1.format("Total simulations:", len(list_kwargs)))
    print(strformat_1_1.format("Number of threads:", n_proc))

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
            print(strformat_1_1_1.format("Simulation progress:", str(int(q.qsize() * 100 / count_total_simulations)), "%"))
            time.sleep(progress_print_interval)
    p.close()
    p.join()
    results = jobs.get()
    time_count_simulation = time.perf_counter() - time_count_simulation
    print(strformat_1_1_1.format("Simulation completed in:", str(int(time_count_simulation)), "s"))

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
                               "PEAK STEEL TEMPERATURE TO FIXED PROTECTION [C]": np.sort(results[:, 11])-273.15,
                               "INDEX": results[:, 12]})
    df_outputs = df_outputs[["TIME EQUIVALENCE [min]", "PEAK STEEL TEMPERATURE TO GOAL SEEK [C]", "PROTECTION THICKNESS [m]", "SEEK STATUS [bool]", "SEEK ITERATIONS [-]", "WINDOW OPEN FRACTION [%]", "FIRE LOAD DENSITY [MJ/m2]", "FIRE SPREAD SPEED [m/s]", "BEAM POSITION [m]", "MAX. NEAR FIELD TEMPERATURE [C]", "FIRE TYPE [0:P., 1:T.]", "PEAK STEEL TEMPERATURE TO FIXED PROTECTION [C]", "INDEX"]]
    df_outputs.set_index("INDEX", inplace=True)
    # df_outputs.sort_values(by=["TIME EQUIVALENCE [min]"], inplace=True)
    # df_outputs.reset_index(drop=True, inplace=True)

    path_results_file = os.path.join(dir_work, "{} - {}".format(id_, "res_df.p"))
    pdump(df_outputs, open(path_results_file, "wb"))
    saveprint(os.path.basename(path_results_file))


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
    saveprint(os.path.basename(path_csv_file))


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

    # Define horizontal line(s) to plot
    if height_building > 0:
        y__ = 1 - 64.8 / height_building ** 2
    else:
        y__ = 0.5

    # Obtain time equivalence, in minutes, as x-axis values
    x = np.sort(df_results["TIME EQUIVALENCE [min]"].values * 60.)
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
                  "legend_is_shown": False,
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

    file_name = "{} - {}{}".format(id_, "res_plot_teq", ".png")
    file_path = os.path.join(dir_work, file_name)
    plt.save_figure2(file_path)
    saveprint(file_name)


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
                  "legend_is_shown": False,
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
            plt.add_text(x=x_end, y=y_line_, s="{:.3f}".format(y_line_), va="left", ha="center", fontsize=6)

    # plt.plot_vertical_line(x=57)
    # plt.add_text(x=57, y=1, s="{}".format("Max. 57"), va="bottom", ha="center", fontsize=6)

    file_name = "{}{}".format(os.path.basename(dir_work), ".png")
    plt.save_figure(dir_folder=dir_work, file_name=file_name, file_format=".png")
    saveprint(file_name)


def step6_results_visualization_temperature(path_input_file):

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

    # Define horizontal line(s) to plot
    # if height_building > 0:
    #     y__ = 1 - 64.8 / height_building ** 2
    # else:
    #     y__ = 0.5

    # Obtain time equivalence, in minutes, as x-axis values
    x = df_results["PEAK STEEL TEMPERATURE TO FIXED PROTECTION [C]"].values
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / len(x)
    # f_interp = interp1d(y, x)
    # if height_building > 0:
    #     y_line = 1 - 64.8 / height_building ** 2
    #     x_line = f_interp(y_line)
    # else:
    #     x_line = y_line = 0

    plt = Scatter2D()
    plt_format = {"figure_size_scale": 0.7,
                  "axis_lim_y1": (0, 1),
                  # "axis_lim_x": (0, 120),
                  "legend_is_shown": False,
                  "axis_label_x": "Peak Steel Temperature [$^\circ$C]",
                  "axis_label_y1": "Fractile",
                  "marker_size": 0}
    plt.plot2(x, y, "Simulation results")

    plt.format(**plt_format)

    file_name = "{} - {}{}".format(id_, "res_plot_temp", ".png")
    file_path = os.path.join(dir_work, file_name)
    plt.save_figure2(file_path)

    saveprint(os.path.basename(file_path))


def step7_select_fires_teq(percentile, dir_work, id_, tolerance=0):

    # Make full path of results and input argument files
    file_name_results = " - ".join([id_, "res_df.p"])
    file_name_input_arguments = " - ".join([id_, "args_main.p"])
    path_results = os.path.join(dir_work, file_name_results)
    path_input_arguments = os.path.join(dir_work, file_name_input_arguments)

    # Load results and input arguments
    df_results = pload(open(path_results, "rb"))
    df_input_arguments = pload(open(path_input_arguments, "rb"))
    # df_all = pd.concat([df_results, df_input_arguments], axis=1)
    # print(df_results.to_string)
    df_results.sort_values(by=["TIME EQUIVALENCE [min]"], inplace=True)
    # print(df_results.to_string)

    if tolerance < 0:
        tolerance = 0

    # Convert 'percentile_ubound' and 'percentile_lbound' to integers according to actual range i.e. 'index=1000'
    # print(df_results.to_string)

    index_max = max(df_results.index.values)

    percentile_ubound = percentile + abs(tolerance)
    percentile_lbound = percentile - abs(tolerance)

    percentile_ubound *= index_max
    percentile_lbound *= index_max

    percentile_ubound = int(round(percentile_ubound, 0))
    percentile_lbound = int(round(percentile_lbound, 0))

    if percentile_lbound <= percentile_ubound:
        range_selected = np.arange(percentile_lbound, percentile_ubound+1, 1)
    else:
        print("NO FIRES ARE SELECTED.")
        return 0

    # df_results.reset_index(drop=True, inplace=True)
    # print(df_results.index.values)
    # print(df_results["TIME EQUIVALENCE [min]"].values)

    # print(masked_range)

    list_index_selected_fires = df_results.iloc[range_selected].index.values
    df_results_selected = df_results.iloc[range_selected]
    df_input_arguments_selected = df_input_arguments.loc[df_results_selected.index.values]

    # iterate through all selected fires, store time and temperature
    plt = Scatter2D()
    dict_fires = {}
    list_fire_name = ["TEMPERATURE {} [C]".format(str(i)) for i,v in enumerate(list_index_selected_fires)]
    # list_time_name = []
    for i,v in enumerate(list_index_selected_fires):
        # get input arguments
        args = df_input_arguments.loc[i].to_dict()

        # get fire type
        fire_type = int(df_results.loc[i]["FIRE TYPE [0:P., 1:T.]"])

        if fire_type == 0:  # parametric fire
            w, l, h = args["room_breadth"], args["room_depth"], args["room_height"]
            inputs_parametric_fire = {
                "A_t": 2*(w*l+w*h+h*l),
                "A_f": w*l,
                "A_v": args["window_height"] * args["window_width"] * args["window_open_fraction"],
                "h_eq": args["window_height"],
                "q_fd": args["fire_load_density"] * 1e6,
                "lambda_": args["room_wall_thermal_inertia"] ** 2,  # thermal inertia is used instead of k rho c.
                "rho": 1,  # see comment for lambda_
                "c": 1,  # see comment for lambda_
                "t_lim": args["time_limiting"],
                "time_end": args["fire_duration"],
                "time_step": args["time_step"],
                "time_start": args["time_start"],
                # "time_padding": (0, 0),
                "temperature_initial": 20 + 273.15,
            }
            tsec, temps = _fire_param(**inputs_parametric_fire)
            # print("")
        elif fire_type == 1:  # travelling fire
            inputs_travelling_fire = {
                "fire_load_density_MJm2": args["fire_load_density"],
                "heat_release_rate_density_MWm2": args["fire_hrr_density"],
                "length_compartment_m": args["room_depth"],
                "width_compartment_m": args["room_breadth"],
                "fire_spread_rate_ms": args["fire_spread_speed"],
                "height_fuel_to_element_m": args["room_height"],
                "length_element_to_fire_origin_m": args["beam_position"],
                "time_start_s": args["time_start"],
                "time_end_s": args["fire_duration"],
                "time_interval_s": args["time_step"],
                "nft_max_C": args["temperature_max_near_field"],
                "win_width_m": args["window_width"],
                "win_height_m": args["window_height"],
                "open_fract": args["window_open_fraction"]
            }
            tsec, temps, hrr, r = _fire_travelling(**inputs_travelling_fire)
            temps += 273.15
            # print("")
        else:
            print("FIRE TYPE UNKOWN.")

        dict_fires[list_fire_name[i]] = temps-273.15
        plt.plot2(tsec/60., temps-273.15, alpha=.6)

    dict_fires["TIME [min]"] = np.arange(args["time_start"],args["fire_duration"],args["time_step"]) / 60.

    df_fires = pd.DataFrame(dict_fires)
    list_names = ["TIME [min]"] + list_fire_name
    df_fires = df_fires[list_names]

    # print(df_fires.to_string)

    # ------------------------------------------------------------------------------------------------------------------
    # Save graphical plot to a .png file
    # ------------------------------------------------------------------------------------------------------------------
    # format parameters for figure
    plt_format = {
        "figure_size_scale": 0.4,
        "axis_lim_y1": (0, 1400),
        "axis_lim_x": (0, 180),
        "legend_is_shown": False,
        "axis_label_x": "Time [min]",
        "axis_label_y1": "Gas Temperature [$^\circ$C]",
        "marker_size": 0,
        "axis_xtick_major_loc": np.arange(0, 181, 20),
        "line_colours": [(0, 0, 0)]
    }
    plt.format(**plt_format)
    file_name = "{} - {}{}".format(id_, "selected_plots", ".png")
    file_path = os.path.join(dir_work, file_name)

    plt.save_figure2(path_file=file_path)
    saveprint(os.path.basename(file_path))

    # ------------------------------------------------------------------------------------------------------------------
    # Save numerical data to a .csv file
    # ------------------------------------------------------------------------------------------------------------------
    file_name = "{} - {}{}".format(id_, "selected_temperatures", ".csv")
    df_fires.to_csv(os.path.join(dir_work, file_name))
    saveprint(file_name)

    file_name = "{} - {}{}".format(id_, "selected_outputs", ".csv")
    df_results_selected.to_csv(os.path.join(dir_work, file_name))
    saveprint(file_name)

    file_name = "{} - {}{}".format(id_, "selected_inputs", ".csv")
    df_input_arguments_selected.to_csv(os.path.join(dir_work, file_name))
    saveprint(file_name)
