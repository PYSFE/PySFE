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
import project.lhs_max_st_temp.ec3_ht as ht

random.seed(123)

def mc_calculation(
    window_height,
    window_width,
    window_open_fraction,
    room_breadth,
    room_depth,
    room_height,
    fire_load_density,
    fire_hrr_density,
    fire_spread_speed,
    time_limiting,
    room_wall_thermal_inertia,
    fire_duration,
    time_step,
    beam_position,
    time_start,
    temperature_max_near_field,
    beam_rho,
    beam_c,
    beam_cross_section_area,
    protection_k,
    protection_rho,
    protection_c,
    protection_depth,
    protection_protected_perimeter
):

    #   Check on applicable fire curve
    av = window_height * window_width * window_open_fraction
    af = room_breadth * room_depth
    at = (2 * af) + ((room_breadth + room_depth) * 2 * room_height)

    #   Opening factor - is it within EC limits?
    of_check = pf.opening_factor(av, window_height, at)

    #   Spread speed - Does the fire spread to involve the full compartment?
    sp_time = max([room_depth, room_breadth]) / fire_spread_speed
    burnout_m2 = max([fire_load_density / fire_hrr_density, 900.])

    if sp_time < burnout_m2 and 0.02 < of_check <= 0.2:  # If fire spreads throughout compartment and ventilation is within EC limits = Parametric fire

        #   Get parametric fire curve
        tsec, tmin, temps = pf.param_fire(room_breadth, room_depth, room_height, window_width, window_height, window_open_fraction, fire_load_density, time_limiting,
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

    #   Solve heat transfer using EC3 correlations
    data_dict_out, time, tempsteel, temperature_rate_steel, specific_heat_steel = \
        ht.make_temperature_eurocode_protected_steel(
            tsec, temps, beam_rho, beam_c, beam_cross_section_area, protection_k, protection_rho, protection_c, protection_depth, protection_protected_perimeter
        )
    max_temp = np.amax(tempsteel)

    return max_temp


def mc_inputs_maker(simulation_count):

    steel_prop = Thermal()
    c = steel_prop.c()
    rho = steel_prop.rho()

    #   Handy interim functions

    def linear_dist(min, max, prob):
        lin_invar = ((max - min) * prob) + min
        return lin_invar

    #   Define the inputs

    lhs_iterations = simulation_count

    #   Compartment dimensions all in [m]

    breadth = 22.4        #   Room breadth [m]
    depth = 44.8          #   Room depth [m]
    height = 3.0        #   Room height [m]
    win_width = 90      #   Window width [m]
    win_height = 2.5    #   Window height [m]

    #   Deterministic fire inputs

    t_start = 0             #   Start time of simulation [s]
    limit_time = 0.333      #   Limiting time for fuel controlled case in EN 1991-1-2 parametric fire [hr]
    inertia = 720           #   Compartment thermal inertia [J/m2s1/2K]
    fire_dur = 18000        #   Maximum time in time array [s]
    time_step = 30          #   Time step used for fire and heat transfer [s]
    hrr_pua = 0.25          #   HRR density [MW/sq.m]

    #   Section properties for heat transfer evaluation

    Hp = 2.14               #   Heated perimeter [m]
    Ap = 0.017              #   Cross section area [sq.m]
    dp = 0.0125             #   Thickness of protection [m]
    kp = 0.2                #   Protection conductivity [W/m.K]
    rhop = 800              #   Density of protection [kg/cb.m]
    cp = 1700               #   Specific heat of protection [J/kg.K]

    #   Set distribution mean and standard dev

    qfd_std = 126           #   Fire load density - Gumbel distribution - standard dev [MJ/sq.m]
    qfd_mean = 420          #   Fire load density - Gumbel distribution - mean [MJ/sq.m]
    glaz_min = 0.1          #   Min glazing fall-out fraction [-] - Linear dist
    glaz_max = 0.999        #   Max glazing fall-out fraction [-]  - Linear dist
    beam_min = 0.6          #   Min beam location relative to compartment length for TFM [-]  - Linear dist
    beam_max = 0.9          #   Max beam location relative to compartment length for TFM [-]  - Linear dist
    com_eff_min = 0.75      #   Min combustion efficiency [-]  - Linear dist
    com_eff_max = 0.999     #   Max combustion efficiency [-]  - Linear dist
    spread_min = 0.0035     #   Min spread rate for TFM [m/s]  - Linear dist
    spread_max = 0.0193     #   Max spread rate for TFM [m/s]  - Linear dist
    avg_nft = 1050          #   TFM near field temperature - Norm distribution - mean [C]

    #   Create random number array for each stochastic variable

    #   Variable 1
    rnd_array_1 = lhs(1, samples=lhs_iterations).flatten()
    rnd_array_1 = np.array(rnd_array_1)
    #   Variable 2
    rnd_array_2 = lhs(1, samples=lhs_iterations).flatten()
    rnd_array_2 = np.array(rnd_array_2)
    #   Variable 3
    rnd_array_3 = lhs(1, samples=lhs_iterations).flatten()
    rnd_array_3 = np.array(rnd_array_3)
    #   Variable 4
    rnd_array_4 = lhs(1, samples=lhs_iterations).flatten()
    rnd_array_4 = np.array(rnd_array_4)
    #   Variable 5
    rnd_array_5 = lhs(1, samples=lhs_iterations).flatten()
    rnd_array_5 = np.array(rnd_array_5)
    #   Variable 6
    rnd_array_6 = lhs(1, samples=lhs_iterations).flatten()
    rnd_array_6 = np.array(rnd_array_6)

    #   Calculate gumbel parameters
    qfd_scale = (qfd_std * (6 ** 0.5)) / np.pi
    qfd_loc = qfd_mean - (0.5722 * qfd_scale)

    #   Near field standard deviation
    std_nft = (1.939 - (np.log(avg_nft) * 0.266)) * avg_nft

    #   Convert LHS probabilities to distribution invariants
    comb_lhs = linear_dist(com_eff_min, com_eff_max, rnd_array_4)
    qfd_lhs = gumbel_r(loc=qfd_loc, scale=qfd_scale).ppf(rnd_array_1) * comb_lhs
    glaz_lhs = linear_dist(glaz_min, glaz_max, rnd_array_2)
    beam_lhs = linear_dist(beam_min, beam_max, rnd_array_3) * depth
    spread_lhs = linear_dist(spread_min, spread_max, rnd_array_5)
    nft_lhs = norm(loc=avg_nft, scale=std_nft).ppf(rnd_array_6)

    list_inputs = []
    for i in range(0, lhs_iterations):
        dict_inputs = {
            "window_height": win_height,
            "window_width": win_width,
            "window_open_fraction": glaz_lhs[i],
            "room_breadth": breadth,
            "room_depth": depth,
            "room_height": height,
            "fire_load_density": qfd_lhs[i],
            "fire_hrr_density": hrr_pua,
            "fire_spread_speed": spread_lhs[i],
            "time_limiting": limit_time,
            "room_wall_thermal_inertia": inertia,
            "fire_duration": fire_dur,
            "time_step": time_step,
            "beam_position": beam_lhs[i],
            "time_start": t_start,
            "temperature_max_near_field": min([nft_lhs[i], 1200]),
            "beam_rho": rho,
            "beam_c": c,
            "beam_cross_section_area": Ap,
            "protection_k": kp,
            "protection_rho": rhop,
            "protection_c": cp,
            "protection_depth": dp,
            "protection_protected_perimeter": Hp
        }
        list_inputs.append(dict_inputs)

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
    is_track_progress = True

    # make all inputs on the one go
    list_kwargs = mc_inputs_maker(simulation_count=1000)

    # Print starting

    print("Starting simulation!")

    # implement of mp, with ability to track progress
    time1 = time.perf_counter()
    if is_track_progress:
        m = mp.Manager()
        q = m.Queue()
        p = mp.Pool(os.cpu_count())
        jobs = p.map_async(worker_with_progress_tracker, [(kwargs, q) for kwargs in list_kwargs])
        count_total_simulations = len(list_kwargs)
        while True:
            if jobs.ready():
                break
            else:
                print("Complete =", q.qsize() * 100 / count_total_simulations, "%")
                time.sleep(2)
        results = jobs.get()
    else:
        pool = mp.Pool(os.cpu_count())
        results = pool.map(worker, list_kwargs)
    time1 = time.perf_counter() - time1
    print("Complete = 100.0 %")
    print("Done! Simulation time =", time1, "s")

    results = np.array(results)
    results.sort()
    percentile = np.arange(1, np.shape(results)[0]+1, 1) / np.shape(results)[0]

    #   Write to csv

    n_rows, = np.shape(results)
    out = np.append(results, percentile, 0)
    out = np.reshape(out, (n_rows, 2), order='F')
    np.savetxt("lhs_mp_out.csv", out, delimiter=",")

    plt.figure(os.cpu_count())
    plt.plot(results, percentile)
    plt.grid(True)
    plt.xlabel('Max Steel Temperature [Deg C]')
    plt.ylabel('Fractile [-]')
    plt.show()
    #
