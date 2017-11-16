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
from project.func.temperature_fires import parametric_eurocode1 as _fire_param


logging.basicConfig(filename="log.txt", level=logging.DEBUG)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


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
        protection_k,
        protection_rho,
        protection_c,
        protection_thickness,
        protection_protected_perimeter,
        temperature_max_near_field=1200,
        index=-1,
):

    #   Check on applicable fire curve
    window_area = window_height * window_width * window_open_fraction
    room_floor_area = room_breadth * room_depth
    room_area = (2 * room_floor_area) + ((room_breadth + room_depth) * 2 * room_height)

    #   Opening factor - is it within EC limits?
    opening_factor = pf.opening_factor(window_area, window_height, room_area)

    #   Spread speed - Does the fire spread to involve the full compartment?
    sp_time = max([room_depth, room_breadth]) / fire_spread_speed
    burnout_m2 = max([fire_load_density / fire_hrr_density, 900.])

    inputs_parametric_fire = {"A_t": room_area,
                              "A_f": room_floor_area,
                              "A_v": window_area,
                              "h_eq": window_height,
                              "q_fd": fire_load_density * 1e6,
                              "lambda_": room_wall_thermal_inertia**2,
                              "rho": 1,
                              "c": 1,
                              "t_lim": time_limiting,
                              "time_end": fire_duration,
                              "time_step": time_step,
                              "time_start": time_start,
                              # "time_padding": (0, 0),
                              "temperature_initial": 20+273.15,}

    if sp_time < burnout_m2 and 0.02 < opening_factor <= 0.2:  # If fire spreads throughout compartment and ventilation is within EC limits = Parametric fire
        #   Get parametric fire curve
        # tsec_, tmin_, temps_ = pf.param_fire(room_breadth, room_depth, room_height, window_width, window_height,
        #                                   window_open_fraction, fire_load_density, time_limiting,
        #                                   room_wall_thermal_inertia, fire_duration,
        #                                   time_step)
        tsec, temps = _fire_param(**inputs_parametric_fire)
        
        fire_type = "Parametric"

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
        temps += 273.15
        fire_type = "Travelling"

    # print("LHS_model_realisation_count =", i + 1, "Of", lhs_iterations)

    #   Optional unprotected steel code
    # tempsteel, temprate, hf, c_s = ht. make_temperature_eurocode_unprotected_steel(tsec,temps+273.15,Hp,beam_cross_section_area,0.1,7850,beam_c,35,0.625)
    # tempsteel -= 273.15
    # max_temp = np.amax(tempsteel)

    # Solve heat transfer using EC3 correlations
    # SI UNITS FOR INPUTS!
    inputs_steel_heat_transfer = {
        "time": tsec,
        "temperature_ambient": temps,
        "rho_steel_T": beam_rho,
        "c_steel_T": beam_c,
        "area_steel_section": beam_cross_section_area,
        "k_protection": protection_k,
        "rho_protection": protection_rho,
        "c_protection": protection_c,
        "thickness_protection": protection_thickness,
        "perimeter_protected": protection_protected_perimeter,
    }
    max_temp = np.max(_steel_temperature(**inputs_steel_heat_transfer)[1])

    return max_temp


def mc_calculation_fr(
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
        dict_time_equivalence=None,
        index="N/A",
):

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
        count_iter_seek = 0
        max_iter_seek = 20
        ubound_seek = 0.5
        lbound_seek = 0.0001
        tol_y_seek = 5
        status_seek = False
        kwargs_steel["thickness_protection"] = (ubound_seek + lbound_seek) / 2
        while count_iter_seek < max_iter_seek:
            count_iter_seek += 1
            max_temp = np.max(_steel_temperature(**kwargs_steel)[1])
            y_diff_seek = max_temp - beam_temperature_max_goal
            if abs(y_diff_seek) <= tol_y_seek:
                status_seek = True
                break
            elif max_temp > beam_temperature_max_goal:
                ubound_seek = kwargs_steel["thickness_protection"]
            else:
                lbound_seek = kwargs_steel["thickness_protection"]
            kwargs_steel["thickness_protection"] = np.average([ubound_seek, lbound_seek])

    else:
        max_temp = np.max(_steel_temperature(**kwargs_steel)[1])

    # TIME-EQUIVALENCE
    if dict_time_equivalence is not None:
        # Make steel time-temperature curve in given fire (i.e. ISO 834)
        kwargs_steel["time"] = dict_time_equivalence["fire_time"]
        kwargs_steel["temperature_ambient"] = dict_time_equivalence["fire_temperature"]
        time, temperature_steel, data_all = _steel_temperature(**kwargs_steel)
        interp_steel_resistance = interp1d(temperature_steel, time)
        if dict_time_equivalence["steel_match_temperature"] == "peak":
            dict_time_equivalence["steel_match_temperature"] = max_temp
        time_equivalence = interp_steel_resistance(400+273.15)
    else:
        time_equivalence = np.nan

    # DEBUG ONLY
    if "count_iter_seek" not in locals():
        status_seek, count_iter_seek, y_diff_seek = "N/A", "N/A", "N/A"

    msg_content = ("INDEX", index,
                   "MINIMISE STATUS", str(status_seek),
                   "Y (Temp. tol.)", y_diff_seek,
                   "N ITER", count_iter_seek,
                   "X (P. thickness)", kwargs_steel["thickness_protection"],
                   "MAX. FIRE TEMP.", np.max(kwargs_steel["temperature_ambient"]),
                   "MAX. STEEL TEMP.", max_temp,
                   "FIRE TYPE", fmstr,
                   "OPENING FACTOR", opening_factor)
    msg = "\n{:18}: {:<5}" * 9
    log.debug("".join([msg.format(*msg_content)]))

    return max_temp, kwargs_steel["thickness_protection"], time_equivalence


def mc_inputs_generator(simulation_count, dict_extra_inputs=None):
    steel_prop = Thermal()

    #   Handy interim functions

    def linear_distribution(min, max, prob): return ((max - min) * prob) + min

    #   Define the inputs

    lhs_iterations = simulation_count
    standard_fire = standard_fire_iso834(np.arange(0, 36000+60, 20), 293.15)

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

    if dict_extra_inputs is not None:
        x.update(dict_extra_inputs)

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
                   "temperature_max_near_field": min([nft_lhs[i], 1200]),
                   "index": i},)
        list_inputs.append(x_)

    return list_inputs