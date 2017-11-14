# -*-coding: utf-8 -*-
import os
import random
import time
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from project.lhs_max_st_temp._lhs_mp_func import mc_calculation, mc_inputs_generator


# wrapper to deal with inputs format (dict-> kwargs)
def worker_with_progress_tracker(arg):
    kwargs, q = arg
    result = mc_calculation(**kwargs)
    q.put(kwargs)
    return result


def worker(arg): return mc_calculation(**arg)


if __name__ == "__main__":
    # SETTINGS
    simulation_count = 100
    progress_print_interval = 5  # [s] 0 to disable progress print
    beam_temperature_max_goal = 473.15  # [K] 0 to disable protection thickness adjustment to peak temperature
    count_process_threads = 0  # 0 to use maximum available processors
    is_time_equivalence = True
    # NOTE: go to function mc_inputs_maker to adjust parameters for the monte carlo simulation

    # SETTING 2
    output_string_start = "{} - START."
    output_string_progress = "Complete = {:3.0f} %."
    output_string_complete = "{} - COMPLETED IN {:0.1f} SECONDS."
    random.seed(123)

    # MAKE INPUTS
    print(output_string_start.format("GENERATE INPUTS"))
    time_count_inputs_maker = time.perf_counter()
    list_kwargs = mc_inputs_generator(simulation_count, beam_temperature_max_goal, is_time_equivalence)
    time_count_inputs_maker = time.perf_counter() - time_count_inputs_maker
    print(output_string_complete.format("GENERATE INPUTS", time_count_inputs_maker))

    # SIMULATION
    print(output_string_start.format("SIMULATION"))
    time_count_simulation = time.perf_counter()
    if progress_print_interval > 0 or count_process_threads < 0:
        # Built-in progress tracking feature enable displaying
        m = mp.Manager()
        q = m.Queue()
        p = mp.Pool(os.cpu_count())
        jobs = p.map_async(worker_with_progress_tracker, [(kwargs, q) for kwargs in list_kwargs])
        count_total_simulations = len(list_kwargs)
        while True:
            if jobs.ready():
                print(output_string_progress.format(100))
                break
            else:
                print(output_string_progress.format(q.qsize() * 100 / count_total_simulations))
                time.sleep(progress_print_interval)
        results = jobs.get()
    else:
        pool = mp.Pool(os.cpu_count())
        results = pool.map(worker, list_kwargs)
    time_count_simulation = time.perf_counter() - time_count_simulation
    print(output_string_complete.format("SIMULATION", time_count_simulation))

    # POST PROCESS
    # format outputs
    results = np.array(results)
    temperature_max_steel = results[:, 0]
    protection_thickness = results[:, 1]
    time_equivalence = results[:, 2]
    protection_thickness = protection_thickness[np.argsort(temperature_max_steel)]
    time_equivalence = time_equivalence[np.argsort(temperature_max_steel)]
    time_equivalence = np.sort(time_equivalence)
    temperature_max_steel = np.sort(temperature_max_steel)
    percentile = np.arange(1, simulation_count + 1) / simulation_count
    pf_outputs = pd.DataFrame({"PERCENTILE [%]": percentile,
                               "PEAK STEEL TEMPERATURE [C]": temperature_max_steel,
                               "PROTECTION LAYER THICKNESS [m]": protection_thickness,
                               "TIME EQUIVALENCE [s]": time_equivalence})

    # write outputs to csv
    pf_outputs.to_csv("output.csv", index=False)

    # plot outputs
    plt.figure(1)
    plt.subplot(211)
    plt.plot(temperature_max_steel, percentile)
    plt.grid(True)
    plt.xlabel('Max Steel Temperature [Deg C]')
    plt.ylabel('Fractile [-]')

    plt.subplot(212)
    plt.plot(time_equivalence/60, percentile)
    plt.grid(True)
    plt.xlabel("Time Equivalence [min]")
    plt.ylabel("Fractile [-]")
    plt.tight_layout()
    plt.show()
