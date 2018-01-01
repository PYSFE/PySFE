# -*- coding: utf-8 -*-
import os
import time
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from project.cls.plot import Scatter2D
from project.lhs_max_st_temp.func import mc_calculation, mc_inputs_generator, mc_post_processing


# wrapper to deal with inputs format (dict-> kwargs)
def worker(arg):
    kwargs, q = arg
    result = mc_calculation(**kwargs)
    q.put(kwargs)
    return result


if __name__ == "__main__":
    # SETTINGS
    simulation_count = 1000
    progress_print_interval = 1  # [s]
    count_process_threads = 0  # 0 to use maximum available processors
    # NOTE: go to function mc_inputs_maker to adjust parameters for the monte carlo simulation

    # MAKE INPUTS
    time_count_inputs_maker = time.perf_counter()
    list_kwargs = mc_inputs_generator(simulation_count)
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

    results = np.array(results)

    # POST PROCESS
    # format outputs
    df_outputs = pd.DataFrame({"PEAK STEEL TEMPERATURE [C]": results[:, 0] - 273.15,
                               "WINDOW OPEN FRACTION [%]": results[:, 1],
                               "FIRE LOAD DENSITY [MJ/m2]": results[:, 2],
                               "FIRE SPREAD SPEED [m/s]": results[:, 3],
                               "BEAM POSITION [m]": results[:, 4],
                               "MAX. NEAR FIELD TEMPERATURE [C]": results[:, 5],
                               "FIRE TYPE [0:P., 1:T.]": results[:, 6]})
    df_outputs = df_outputs[["PEAK STEEL TEMPERATURE [C]", "WINDOW OPEN FRACTION [%]", "FIRE LOAD DENSITY [MJ/m2]", "FIRE SPREAD SPEED [m/s]", "BEAM POSITION [m]", "MAX. NEAR FIELD TEMPERATURE [C]", "FIRE TYPE [0:P., 1:T.]"]]
    df_outputs.sort_values("PEAK STEEL TEMPERATURE [C]", inplace=True)
    df_outputs.reset_index(drop=True, inplace=True)

    # write outputs to csv
    df_outputs.to_csv("results_numerical.csv", index=True)

    # plot outputs
    x = results[:, 0]
    x, y, x_, y_, c = mc_post_processing(x)
    plt = Scatter2D()
    plt.plot2(x-273.15, y, "Simulation results")
    plt.plot2(x_-273.15, y_, "Interpreted CDF")
    plt.format(**{"figure_size_scale": 0.5,
                  "legend_is_shown": True,
                  "axis_label_x": "Max Steel Temperature [$\degree C$]",
                  "axis_label_y1": "Fractile",
                  "marker_size": 0})
    plt.update_line_format("Interpreted CDF", **{"line_width": 0.5, "color": "black", "line_style": ":"})
    plt.update_legend()
    plt.save_figure(file_name="results_plot", file_format=".png")
