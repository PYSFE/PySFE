import os
import random
import time
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from project.lhs_max_st_temp.func import mc_calculation, mc_inputs_generator


# wrapper to deal with inputs format (dict-> kwargs)
def worker_with_progress_tracker(arg):
    kwargs, q = arg
    result = mc_calculation(**kwargs)
    q.put(kwargs)
    return result


def worker(arg): return mc_calculation(**arg)


if __name__ == "__main__":
    # SETTINGS
    simulation_count = 1000
    progress_print_interval = 5  # [s]
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
    jobs = p.map_async(worker_with_progress_tracker, [(kwargs, q) for kwargs in list_kwargs])
    count_total_simulations = len(list_kwargs)
    while True:
        if jobs.ready():
            print("SIMULATION COMPLETED {:3.0f} %...".format(100))
            break
        else:
            print("SIMULATION COMPLETED {:3.0f} %...".format(q.qsize() * 100 / count_total_simulations))
            time.sleep(progress_print_interval)
    results = jobs.get()
    time_count_simulation = time.perf_counter() - time_count_simulation
    print("SIMULATION COMPLETED IN {:0.3f} SECONDS".format(time_count_simulation))

    # POST PROCESS
    # format outputs
    results = np.array(results)
    temperature_max_steel = results
    temperature_max_steel = np.sort(temperature_max_steel)
    percentile = np.arange(1, simulation_count + 1) / simulation_count
    pf_outputs = pd.DataFrame({"PERCENTILE [%]": percentile,
                               "PEAK STEEL TEMPERATURE [C]": temperature_max_steel-273.15,})

    # write outputs to csv
    pf_outputs.to_csv("output.csv", index=False)

    # plot outputs
    plt.figure(1)
    plt.subplot(111)
    plt.plot(temperature_max_steel-273.15, percentile)
    plt.grid(True)
    plt.xlabel('Max Steel Temperature [Deg C]')
    plt.ylabel('Fractile [-]')
    plt.show()
