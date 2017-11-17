# -*- coding: utf-8 -*-
import os
import time
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from project.lhs_max_st_temp.func import mc_fr_calculation, mc_inputs_generator
from project.func.temperature_fires import standard_fire_iso834 as standard_fire


# wrapper to deal with inputs format (dict-> kwargs)
def worker_with_progress_tracker(arg):
    kwargs, q = arg
    result = mc_fr_calculation(**kwargs)
    q.put(kwargs)
    return result


if __name__ == "__main__":
    # SETTINGS
    simulation_count = 500
    progress_print_interval = 5  # [s]
    count_process_threads = 0  # 0 to use maximum available processors
    steel_temperature_goal = 273.15+600
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
    jobs = p.map_async(worker_with_progress_tracker, [(kwargs, q) for kwargs in list_kwargs])
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
    results = np.array(results)
    time_equivalence = results
    time_equivalence = np.sort(time_equivalence)
    percentile = np.arange(1, simulation_count + 1) / simulation_count
    df_outputs = pd.DataFrame({"PERCENTILE [%]": percentile,
                               "TIME EQUIVALENCE [min]": time_equivalence/60., })
    df_outputs = df_outputs[["PERCENTILE [%]", "TIME EQUIVALENCE [min]"]]

    # write outputs to csv
    df_outputs.to_csv("output.csv", index=False)

    # plot outputs
    plt.figure(1)
    plt.subplot(111)
    plt.plot(time_equivalence/60., percentile)
    plt.grid(True)
    plt.xlabel('Time Equivalence in ISO 834 [min]')
    plt.ylabel('Fractile [-]')
    plt.show()
