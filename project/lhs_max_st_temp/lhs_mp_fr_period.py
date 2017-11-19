# -*- coding: utf-8 -*-
import os
import time
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from project.lhs_max_st_temp.func import mc_fr_calculation, mc_inputs_generator
from project.func.temperature_fires import standard_fire_iso834 as standard_fire
from project.cls.plot import Scatter2D


# wrapper to deal with inputs format (dict-> kwargs)
def worker(arg):
    kwargs, q = arg
    result = mc_fr_calculation(**kwargs)
    q.put(kwargs)
    return result


if __name__ == "__main__":
    # SETTINGS
    simulation_count = 50
    progress_print_interval = 5  # [s]
    count_process_threads = 2  # 0 to use maximum available processors
    steel_temperature_goal = 273.15+620
    # NOTE: go to function mc_inputs_maker to adjust parameters for the monte carlo simulation

    # MAKE INPUTS
    time_count_inputs_maker = time.perf_counter()
    fire = standard_fire(np.arange(0, 5*60*60, 20), 273.15+20)
    inputs_extra = {"beam_temperature_goal": steel_temperature_goal,
                    "iso834_time": fire[0],
                    "iso834_temperature": fire[1],}
    list_kwargs = mc_inputs_generator(simulation_count, inputs_extra)
    time_count_inputs_maker = time.perf_counter() - time_count_inputs_maker

    # SIMULATION
    print("SIMULATION START")
    time_count_simulation = time.perf_counter()
    m = mp.Manager()
    q = m.Queue()
    p = mp.Pool(os.cpu_count())
    jobs = p.map_async(worker, [(kwargs, q) for kwargs in list_kwargs])
    count_total_simulations = len(list_kwargs)
    while progress_print_interval:
        if jobs.ready():
            break
        else:
            print("SIMULATION COMPLETED {:3.0f} %...".format(q.qsize() * 100 / count_total_simulations))
            time.sleep(progress_print_interval)
    results = jobs.get()
    time_count_simulation = time.perf_counter() - time_count_simulation
    print("SIMULATION COMPLETED IN {:0.3f} SECONDS".format(time_count_simulation, progress_print_interval))

    # POST PROCESS
    # format outputs
    results = np.array(results, dtype=float)
    seek_successful = sum(results[:, 1])
    time_equivalence = results[:, 0][results[:, 1] == True]
    time_equivalence = np.sort(time_equivalence)
    percentile = np.arange(1, seek_successful + 1) / seek_successful
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
    df_outputs.to_csv("results_numerical.csv", index=True)

    # plot outputs
    plt = Scatter2D()
    plt.plot2(time_equivalence/60., percentile)
    plt.format(**{"figure_size_scale": 0.5,
                  "axis_lim_y1": (0, 1),
                  "legend_is_shown": False,
                  "axis_label_x": "Max Steel Temperature [min]",
                  "axis_label_y1": "Fractile",
                  "marker_size": 0})
    plt.save_figure(name="results_plot")
