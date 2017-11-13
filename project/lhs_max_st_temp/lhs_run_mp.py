# -*-coding: utf-8 -*-
import os
import random
import time
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.distributions import norm, gumbel_r
from pyDOE import lhs
from project.dat.steel_carbon import Thermal
import project.lhs_max_st_temp.ec_param_fire as pf
import project.lhs_max_st_temp.tfm_alt as tfma
from project.func.temperature_steel_section import protected_steel_eurocode as _steel_temperature
from scipy.optimize import fmin, newton, minimize, minimize_scalar
import pandas as pd
import copy
import logging
from project.func.temperature_fires import standard_fire_iso834
from scipy.interpolate import interp1d


printd = logging.getLogger(__name__)
printd.setLevel(logging.DEBUG)


def mc_calculation(
        time_step,
        time_start,
        time_limiting,
        window_height,
        window_width,
        window_open_fraction,
        room_breadth,
        room_depth,
        room_height,
        room_wall_thermal_inertia,
        fire_load_density,
        fire_hrr_density,
        fire_spread_speed,
        fire_duration,
        beam_position,
        beam_rho,
        beam_c,
        beam_cross_section_area,
        beam_temperature_max_goal,
        protection_k,
        protection_rho,
        protection_c,
        protection_thickness,
        protection_protected_perimeter,
        temperature_max_near_field=1200,
        beam_protection_thickness_goal_tol=1,
        dict_time_equivalence=None
):
    dict_time_equivalence["status"]
    dict_time_equivalence["fire_time"]
    dict_time_equivalence["fire_temperature"]
    dict_time_equivalence["match_temperature"]

    #   Check on applicable fire curve
    window_area = window_height * window_width * window_open_fraction
    room_floor_area = room_breadth * room_depth
    room_area = (2 * room_floor_area) + ((room_breadth + room_depth) * 2 * room_height)

    #   Opening factor - is it within EC limits?
    opening_factor = pf.opening_factor(window_area, window_height, room_area)

    #   Spread speed - Does the fire spread to involve the full compartment?
    sp_time = max([room_depth, room_breadth]) / fire_spread_speed
    burnout_m2 = max([fire_load_density / fire_hrr_density, 900.])

    if sp_time < burnout_m2 and 0.02 < opening_factor <= 0.2:  # If fire spreads throughout compartment and ventilation is within EC limits = Parametric fire

        #   Get parametric fire curve
        tsec, tmin, temps = pf.param_fire(room_breadth, room_depth, room_height, window_width, window_height,
                                          window_open_fraction, fire_load_density, time_limiting,
                                          room_wall_thermal_inertia, fire_duration,
                                          time_step)
        fmstr = "Parametric"

    else:  # Otherwise, it is a travelling fire

        #   Get travelling fire curve
        tsec, temps, heat_release, distance_to_element = tfma.travelling_fire(
            fire_load_density,
            fire_hrr_density,
            room_depth,
            room_breadth,
            fire_spread_speed,
            room_height,
            beam_position,
            time_start,
            fire_duration,
            time_step,
            temperature_max_near_field,
            window_width,
            window_height,
            window_open_fraction
        )
        fmstr = "Travelling"

    # print("LHS_model_realisation_count =", i + 1, "Of", lhs_iterations)

    #   Optional unprotected steel code
    # tempsteel, temprate, hf, c_s = ht. make_temperature_eurocode_unprotected_steel(tsec,temps+273.15,Hp,beam_cross_section_area,0.1,7850,beam_c,35,0.625)
    # tempsteel -= 273.15
    # max_temp = np.amax(tempsteel)

    # Solve heat transfer using EC3 correlations
    # SI UNITS FOR INPUTS!
    kwargs_steel = {
        "time": tsec,
        "temperature_ambient": temps + 273.15,
        "rho_steel_T": beam_rho,
        "c_steel_T": beam_c,
        "area_steel_section": beam_cross_section_area,
        "k_protection": protection_k,
        "rho_protection": protection_rho,
        "c_protection": protection_c,
        "thickness_protection": protection_thickness,
        "perimeter_protected": protection_protected_perimeter,
    }
    if beam_temperature_max_goal != 0:
        def __steel_temperature(protection_thickness):
            kwargs_steel["thickness_protection"] = protection_thickness
            max_temp_diff = abs(np.max(_steel_temperature(**kwargs_steel)[1]) - beam_temperature_max_goal)
            return max_temp_diff

        res_min = minimize_scalar(fun=__steel_temperature,
                                  method="golden",
                                  bracket=(1e-3, 0.1),
                                  options={"xtol": 1e-4, "maxiter": 20})
        protection_thickness_ = res_min.x
        kwargs_steel["thickness_protection"] = protection_thickness_
        temperature_steel = _steel_temperature(**kwargs_steel)[1]
        max_temp = np.amax(temperature_steel) - 273.15

        printd.debug("".join(["-" * 37, ("\n{:30}: {:<5}" * 5).format("MINIMISE STATUS", str(res_min.success),
                                                                      "DIFF", res_min.fun,
                                                                      "N ITER", res_min.nit,
                                                                      "PROTECTION THICKNESS", protection_thickness_,
                                                                      "MAXIMUM STEEL TEMPERATURE", max_temp)]))
    else:
        protection_thickness_ = protection_thickness
        time, temperature_steel, data_all = _steel_temperature(**kwargs_steel)
        max_temp = np.amax(temperature_steel) - 273.15

    # TIME-EQUIVALENCE
    if dict_time_equivalence is not None:
        # Make steel time-temperature curve in given fire (i.e. ISO 834)
        kwargs_steel["time"] = dict_time_equivalence["fire_time"]
        kwargs_steel["temperature"] = dict_time_equivalence["fire_temperature"]
        steel_heated_iso834 = _steel_temperature(**kwargs_steel)
        interp_steel_resistance = interp1d(steel_heated_iso834[1], steel_heated_iso834[0])
        if dict_time_equivalence["match"] == "peak": dict_time_equivalence["match_temperature"] = max_temp
        equivalence_resistance = interp_steel_resistance(dict_time_equivalence["match_temperature"])
    else:
        equivalence_resistance = np.nan

    return max_temp, protection_thickness_, equivalence_resistance


def mc_inputs_maker_2(simulation_count, beam_temperature_max_goal):
    steel_prop = Thermal()

    #   Handy interim functions

    def linear_distribution(min, max, prob): return ((max - min) * prob) + min

    #   Define the inputs

    lhs_iterations = simulation_count
    standard_fire = standard_fire_iso834(np.arange(0, 36000+60, 60), 293.15)

    #   Compartment dimensions all in [m]
    x = dict()

    x["beam_c"] = steel_prop.c()
    x["beam_rho"] = steel_prop.rho()
    x["room_breadth"] = 22.4  # Room breadth [m]
    x["room_depth"] = 44.8  # Room depth [m]
    x["room_height"] = 3.0  # Room height [m]
    x["window_width"] = 90  # Window width [m]
    x["window_height"] = 2.5  # Window height [m]

    #   Deterministic fire inputs

    x["time_start"] = 0  # Start time of simulation [s]
    x["time_limiting"] = 0.333  # Limiting time for fuel controlled case in EN 1991-1-2 parametric fire [hr]
    x["room_wall_thermal_inertia"] = 720  # Compartment thermal inertia [J/m2s1/2K]
    x["fire_duration"] = 18000  # Maximum time in time array [s]
    x["time_step"] = 30  # Time step used for fire and heat transfer [s]
    x["fire_hrr_density"] = 0.25  # HRR density [MW/sq.m]

    #   Section properties for heat transfer evaluation

    x["protection_protected_perimeter"] = 2.14  # Heated perimeter [m]
    x["beam_cross_section_area"] = 0.017  # Cross section area [sq.m]
    x["protection_thickness"] = 0.0125  # Thickness of protection [m]
    x["protection_k"] = 0.2  # Protection conductivity [W/m.K]
    x["protection_rho"] = 800  # Density of protection [kg/cb.m]
    x["protection_c"] = 1700  # Specific heat of protection [J/kg.K]

    x["beam_temperature_max_goal"] = beam_temperature_max_goal

    x["equivalence_fire_time"] = standard_fire[0]
    x["equivalence_fire_temperature"] = standard_fire[1]
    x["equivalence_steel_temperature"] = "peak",

    #   Set distribution mean and standard dev

    qfd_std = 126  # Fire load density - Gumbel distribution - standard dev [MJ/sq.m]
    qfd_mean = 420  # Fire load density - Gumbel distribution - mean [MJ/sq.m]
    glaz_min = 0.1  # Min glazing fall-out fraction [-] - Linear dist
    glaz_max = 0.999  # Max glazing fall-out fraction [-]  - Linear dist
    beam_min = 0.6  # Min beam location relative to compartment length for TFM [-]  - Linear dist
    beam_max = 0.9  # Max beam location relative to compartment length for TFM [-]  - Linear dist
    com_eff_min = 0.75  # Min combustion efficiency [-]  - Linear dist
    com_eff_max = 0.999  # Max combustion efficiency [-]  - Linear dist
    spread_min = 0.0035  # Min spread rate for TFM [m/s]  - Linear dist
    spread_max = 0.0193  # Max spread rate for TFM [m/s]  - Linear dist
    avg_nft = 1050  # TFM near field temperature - Norm distribution - mean [C]

    #   Create random number array for each stochastic variable

    # #   Variable 1
    # rnd_array_1 = lhs(1, samples=lhs_iterations).flatten()
    # rnd_array_1 = np.array(rnd_array_1)
    # #   Variable 2
    # rnd_array_2 = lhs(1, samples=lhs_iterations).flatten()
    # rnd_array_2 = np.array(rnd_array_2)
    # #   Variable 3
    # rnd_array_3 = lhs(1, samples=lhs_iterations).flatten()
    # rnd_array_3 = np.array(rnd_array_3)
    # #   Variable 4
    # rnd_array_4 = lhs(1, samples=lhs_iterations).flatten()
    # rnd_array_4 = np.array(rnd_array_4)
    # #   Variable 5
    # rnd_array_5 = lhs(1, samples=lhs_iterations).flatten()
    # rnd_array_5 = np.array(rnd_array_5)
    # #   Variable 6
    # rnd_array_6 = lhs(1, samples=lhs_iterations).flatten()
    # rnd_array_6 = np.array(rnd_array_6)

    list_random_numbers = [lhs(1, samples=lhs_iterations).flatten() for r in range(6)]

    #   Calculate gumbel parameters
    qfd_scale = (qfd_std * (6 ** 0.5)) / np.pi
    qfd_loc = qfd_mean - (0.5722 * qfd_scale)

    #   Near field standard deviation
    std_nft = (1.939 - (np.log(avg_nft) * 0.266)) * avg_nft

    #   Convert LHS probabilities to distribution invariants
    comb_lhs = linear_distribution(com_eff_min, com_eff_max, list_random_numbers[0])
    qfd_lhs = gumbel_r(loc=qfd_loc, scale=qfd_scale).ppf(list_random_numbers[1]) * comb_lhs
    glaz_lhs = linear_distribution(glaz_min, glaz_max, list_random_numbers[2])
    beam_lhs = linear_distribution(beam_min, beam_max, list_random_numbers[3]) * x["room_depth"]
    spread_lhs = linear_distribution(spread_min, spread_max, list_random_numbers[4])
    nft_lhs = norm(loc=avg_nft, scale=std_nft).ppf(list_random_numbers[5])

    list_inputs = []
    for i in range(0, lhs_iterations):
        x_ = x.copy()
        x_.update({"window_open_fraction": glaz_lhs[i],
                   "fire_load_density": qfd_lhs[i],
                   "fire_spread_speed": spread_lhs[i],
                   "beam_position": beam_lhs[i],
                   "temperature_max_near_field": min([nft_lhs[i], 1200])})
        list_inputs.append(x_)

    return list_inputs


# wrapper to deal with inputs format (dict-> kwargs)
def worker_with_progress_tracker(arg):
    kwargs, q = arg
    result = mc_calculation(**kwargs)
    q.put(kwargs)
    return result


def worker(arg): return mc_calculation(**arg)


if __name__ == "__main__":
    # SETTINGS
    simulation_count = 1
    progress_print_interval = 2  # [s] 0 to disable progress print
    beam_temperature_max_goal = 873  # [K] 0 to disable protection thickness adjustment to peak temperature
    count_process_threads = 1  # 0 to use maximum available processors
    # NOTE: go to function mc_inputs_maker to adjust parameters for the monte carlo simulation

    # SETTING 2
    output_string_start = "{} - START."
    output_string_progress = "Complete = {:3.0f} %."
    output_string_complete = "{} - COMPLETED IN {:0.1f} SECONDS."
    random.seed(123)

    # make all inputs on the one go
    print(output_string_start.format("GENERATE INPUTS"))
    time_count_inputs_maker = time.perf_counter()
    # list_kwargs = mc_inputs_maker(simulation_count=simulation_count,
    #                               beam_temperature_max_goal=beam_temperature_max_goal)
    list_kwargs = mc_inputs_maker_2(simulation_count, beam_temperature_max_goal)
    time_count_inputs_maker = time.perf_counter() - time_count_inputs_maker
    print(output_string_complete.format("GENERATE INPUTS", time_count_inputs_maker))

    # Print starting

    print(output_string_start.format("SIMULATION"))

    # implement of mp, with ability to track progress
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
    protection_thickness = protection_thickness[np.argsort(temperature_max_steel)]
    temperature_max_steel = np.sort(temperature_max_steel)
    percentile = np.arange(1, simulation_count + 1) / simulation_count
    pf_outputs = pd.DataFrame({"PERCENTILE [%]": percentile,
                               "PEAK STEEL TEMPERATURE [C]": temperature_max_steel,
                               "PROTECTION LAYER THICKNESS [m]": protection_thickness})

    # write outputs to csv
    pf_outputs.to_csv("output.csv", index=False)

    # plot outputs
    plt.figure()
    plt.plot(temperature_max_steel, percentile)
    plt.grid(True)
    plt.xlabel('Max Steel Temperature [Deg C]')
    plt.ylabel('Fractile [-]')
    plt.show()
