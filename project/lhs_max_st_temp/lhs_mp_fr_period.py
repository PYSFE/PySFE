# -*- coding: utf-8 -*-
import os
import time
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from project.lhs_max_st_temp.func import mc_fr_calculation, mc_inputs_generator, mc_post_processing
from project.func.temperature_fires import standard_fire_iso834 as standard_fire
from project.cls.plot import Scatter2D


# wrapper to deal with inputs format (dict-> kwargs)
def worker(arg):
    kwargs, q = arg
    result = mc_fr_calculation(**kwargs)
    q.put(kwargs)
    return result


def main(dir_work, n_sim, T_s_fix, name_in_f="inputs.txt", n_proc=0):
    progress_print_interval = 5  # [s]

    # MAKE INPUTS
    time_count_inputs_maker = time.perf_counter()
    fire = standard_fire(np.arange(0, 3*60*60, 1), 273.15+20)
    inputs_extra = {"beam_temperature_goal": T_s_fix,
                    "iso834_time": fire[0],
                    "iso834_temperature": fire[1],}
    list_kwargs = mc_inputs_generator(n_sim, inputs_extra, dir_file="/".join([dir_work, name_in_f]))
    time_count_inputs_maker = time.perf_counter() - time_count_inputs_maker

    # SIMULATION
    print("SIMULATION STARTS")
    n_proc = os.cpu_count() if int(n_proc) < 1 else int(n_proc)
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
                               "PROTECTION THICKNESS [m]": results[:, 9]})
    df_outputs = df_outputs[["TIME EQUIVALENCE [min]", "PEAK STEEL TEMPERATURE [C]", "PROTECTION THICKNESS [m]", "SEEK STATUS [bool]", "WINDOW OPEN FRACTION [%]", "FIRE LOAD DENSITY [MJ/m2]", "FIRE SPREAD SPEED [m/s]", "BEAM POSITION [m]", "MAX. NEAR FIELD TEMPERATURE [C]", "FIRE TYPE [0:P., 1:T.]"]]
    df_outputs.sort_values(by=["TIME EQUIVALENCE [min]"], inplace=True)
    df_outputs.reset_index(drop=True, inplace=True)

    # write outputs to csv
    df_outputs.to_csv("/".join([dir_work, "results_numerical.csv"]), index=True)

    # plot outputs
    seek_successful = sum(results[:, 1])
    x = results[:, 0][results[:, 1] == True]
    x = np.sort(x)


def post_processing_plot(dir_work, height_building):
    df_results = pd.read_csv("/".join([dir_work, "results_numerical.csv"]))
    x = df_results["TIME EQUIVALENCE [min]"].values * 60.
    seek_status = df_results["SEEK STATUS [bool]"].values
    x = x[seek_status == 1]

    resistance_criteria = 1 - 64.8 / height_building ** 2

    x, y, x_, y_, xy_found = mc_post_processing(x, y_find=[resistance_criteria])
    plt = Scatter2D()
    plt.plot2(x/60, y, "Simulation results")
    plt.plot2(x_/60, y_, "Interpolated CDF")
    plt.format(**{"figure_size_scale": 0.7, "axis_lim_y1": (0, 1), "axis_lim_x": (0, 180), "legend_is_shown": True, "axis_label_x": "Time Equivalence [min]", "axis_label_y1": "Fractile", "marker_size": 0})

    plt.update_line_format("Interpolated CDF", **{"line_width": 0.5, "color": "black", "line_style": ":"})
    x_found, y_found = xy_found[0, :]
    plt.axes_primary.axvline(x=x_found/60., color="black", linewidth=1)
    plt.axes_primary.axhline(y=y_found, color="black", linewidth=1)
    plt.axes_primary.text(x=x_found/60+1, y=y_found-0.01, s="({:.0f}, {:.4f})".format(x_found/60, y_found), va="top", ha="left", fontsize=6)
    plt.update_legend(legend_loc=0)
    plt.save_figure(file_name="results_plot", file_format=".pdf", dir_folder=dir_work)

    print("POST PROCESSING COMPLETE")


if __name__ == "__main__":
    # SETTINGS
    simulations = 50
    steel_temperature_to_fix = 273.15 + 620
    building_height = 66
    project_full_path = "C:/Users/Ian Fu/Dropbox (OFR-UK)/Bicester_team_projects/Live_projects/Symons House/Time Equivalence Analysis/cinema"
    # NOTE: go to function mc_inputs_maker to adjust parameters for the monte carlo simulation

    main(project_full_path, simulations, steel_temperature_to_fix)
    post_processing_plot(project_full_path, building_height)
