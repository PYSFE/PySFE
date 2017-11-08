# -*-coding: utf-8 -*-

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
    import project.lhs_max_st_temp.ec_param_fire as pf
    import project.lhs_max_st_temp.tfm_alt as tfma
    import project.lhs_max_st_temp.ec3_ht as ht
    import numpy as np

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


def mc_inputs_maker():
    from scipy.stats.distributions import norm, gumbel_r, gumbel_l
    import numpy as np
    from project.dat.steel_carbon import Thermal
    from pyDOE import lhs

    steel_prop = Thermal()
    c = steel_prop.c()
    rho = steel_prop.rho()

    #   Handy interim functions

    def linear_dist(min, max, prob):
        lin_invar = ((max - min) * prob) + min
        return lin_invar

    #   Define the inputs

    lhs_iterations = 100

    #   Compartment dimensions

    breadth = 16
    depth = 20
    height = 3.0
    win_width = 20
    win_height = 2.5

    #   Deterministic fire inputs

    t_start = 0
    limit_time = 0.333
    inertia = 720
    fire_dur = 18000
    time_step = 30
    hrr_pua = 0.25

    #   Section properties

    Hp = 2.14
    Ap = 0.017
    dp = 0.0125
    kp = 0.2
    rhop = 800
    cp = 1700

    #   Set distribution mean and standard dev

    qfd_std = 126
    qfd_mean = 420
    glaz_min = 0.1
    glaz_max = 0.999
    beam_min = 0.6
    beam_max = 0.9
    com_eff_min = 0.75
    com_eff_max = 0.999
    spread_min = 0.0035
    spread_max = 0.0193
    avg_nft = 1050

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
def worker(kwargs): return mc_calculation(**kwargs)


if __name__ == "__main__":
    import multiprocessing as mp
    import time

    # make all inputs on the one go
    args = mc_inputs_maker()

    # implement of mp
    time1 = time.perf_counter()
    pool = mp.Pool(2)
    results = pool.map(worker, args)
    time1 = time.perf_counter() - time1

    print(time1)
