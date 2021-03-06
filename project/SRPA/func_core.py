# -*- coding: utf-8 -*-
import copy
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from scipy.stats.distributions import norm, gumbel_r
from pandas import DataFrame as df

import project.lhs_max_st_temp.ec_param_fire as pf
import project.SRPA.tfm_alt as _fire_travelling
from project.dat.steel_carbon import Thermal
from project.func.temperature_steel_section import protected_steel_eurocode as _steel_temperature
from project.func.temperature_fires import parametric_eurocode1 as _fire_param
from project.func.kwargs_from_text import kwargs_from_text
from project.func.probabilistic_latin_hypercube import latin_hypercube_sampling

#
# def mc_calculation(
#         time_step,
#         time_start,
#         time_limiting,
#         window_height,
#         window_width,
#         window_open_fraction,
#         room_breadth,
#         room_depth,
#         room_height,
#         room_wall_thermal_inertia,
#         fire_load_density,
#         fire_hrr_density,
#         fire_spread_speed,
#         fire_duration,
#         beam_position,
#         beam_rho,
#         beam_c,
#         beam_cross_section_area,
#         protection_k,
#         protection_rho,
#         protection_c,
#         protection_thickness,
#         protection_protected_perimeter,
#         temperature_max_near_field=1200,
#         index=-1,
# ):
#
#     #   Check on applicable fire curve
#     window_area = window_height * window_width * window_open_fraction
#     room_floor_area = room_breadth * room_depth
#     room_area = (2 * room_floor_area) + ((room_breadth + room_depth) * 2 * room_height)
#
#     #   Opening factor - is it within EC limits?
#     opening_factor = pf.opening_factor(window_area, window_height, room_area)
#
#     #   Spread speed - Does the fire spread to involve the full compartment?
#     sp_time = max([room_depth, room_breadth]) / fire_spread_speed
#     burnout_m2 = max([fire_load_density / fire_hrr_density, 900.])
#
#     inputs_parametric_fire = {"A_t": room_area,
#                               "A_f": room_floor_area,
#                               "A_v": window_area,
#                               "h_eq": window_height,
#                               "q_fd": fire_load_density * 1e6,
#                               "lambda_": room_wall_thermal_inertia**2,  # thermal inertia is used instead of k rho c.
#                               "rho": 1,
#                               "c": 1,
#                               "t_lim": time_limiting,
#                               "time_end": fire_duration,
#                               "time_step": time_step,
#                               "time_start": time_start,
#                               # "time_padding": (0, 0),
#                               "temperature_initial": 20+273.15,}
#
#     # If fire spreads throughout compartment and ventilation is within EC limits = Parametric fire
#     # Otherwise, it is a travelling fire
#     if sp_time < burnout_m2 and 0.02 < opening_factor <= 0.2:
#         #   Get parametric fire curve
#         tsec, temps = _fire_param(**inputs_parametric_fire)
#
#         fire_type = 0
#
#     else:
#         #   Get travelling fire curve
#         tsec, temps, heat_release, distance_to_element = _fire_travelling.travelling_fire(
#             fire_load_density,
#             fire_hrr_density,
#             room_depth,
#             room_breadth,
#             fire_spread_speed,
#             room_height,
#             beam_position,
#             time_start,
#             fire_duration,
#             time_step,
#             temperature_max_near_field,
#             window_width,
#             window_height,
#             window_open_fraction
#         )
#         temps += 273.15
#         fire_type = 1
#
#     #   Optional unprotected steel code
#     # tempsteel, temprate, hf, c_s = ht. make_temperature_eurocode_unprotected_steel(tsec,temps+273.15,Hp,beam_cross_section_area,0.1,7850,beam_c,35,0.625)
#     # tempsteel -= 273.15
#     # max_temp = np.amax(tempsteel)
#
#     # Solve heat transfer using EC3 correlations
#     # SI UNITS FOR INPUTS!
#     inputs_steel_heat_transfer = {
#         "time": tsec,
#         "temperature_ambient": temps,
#         "rho_steel": beam_rho,
#         "c_steel_T": beam_c,
#         "area_steel_section": beam_cross_section_area,
#         "k_protection": protection_k,
#         "rho_protection": protection_rho,
#         "c_protection": protection_c,
#         "thickness_protection": protection_thickness,
#         "perimeter_protected": protection_protected_perimeter,
#     }
#     max_temp = np.max(_steel_temperature(**inputs_steel_heat_transfer)[1])
#
#     return max_temp, window_open_fraction, fire_load_density, fire_spread_speed, beam_position, temperature_max_near_field, fire_type, index
#

def calc_time_equiv_worker(arg):
    kwargs, q = arg
    result = calculation_time_equivalence(**kwargs)
    q.put("index: {}".format(kwargs["index"]))
    return result


def calculation_time_equivalence(
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
        beam_temperature_goal,
        protection_k,
        protection_rho,
        protection_c,
        protection_thickness,
        protection_protected_perimeter,
        iso834_time,
        iso834_temperature,
        temperature_max_near_field=1200,
        seek_max_iter=20,
        seek_ubound=0.1,
        seek_lbound=0.0001,
        seek_tol_y=1,
        index=-1,
        **kwargs,
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
        tsec, temps = _fire_param(**inputs_parametric_fire)

        fire_type = 0

    else:  # Otherwise, it is a travelling fire
        #   Get travelling fire curve
        tsec, temps, heat_release, distance_to_element = _fire_travelling.travelling_fire(
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
            window_open_fraction,
        )
        temps += 273.15
        fire_type = 1

    #   Optional unprotected steel code
    # tempsteel, temprate, hf, c_s = ht. make_temperature_eurocode_unprotected_steel(tsec,temps+273.15,Hp,beam_cross_section_area,0.1,7850,beam_c,35,0.625)
    # tempsteel -= 273.15
    # max_temp = np.amax(tempsteel)

    # Solve heat transfer using EC3 correlations
    # SI UNITS FOR INPUTS!
    inputs_steel_heat_transfer = {"time": tsec,
                                  "temperature_ambient": temps,
                                  "rho_steel": beam_rho,
                                  "c_steel_T": beam_c,
                                  "area_steel_section": beam_cross_section_area,
                                  "k_protection": protection_k,
                                  "rho_protection": protection_rho,
                                  "c_protection": protection_c,
                                  "thickness_protection": protection_thickness,
                                  "perimeter_protected": protection_protected_perimeter,
                                  "is_terminate_peak": True}

    # Find maximum steel temperature for the static protection layer thickness
    if protection_thickness > 0:
        temperature_steel_max = np.max(_steel_temperature(**inputs_steel_heat_transfer)[1])
    else:
        temperature_steel_max = -1

    # MATCH PEAK STEEL TEMPERATURE BY ADJUSTING PROTECTION LAYER THICKNESS
    seek_count_iter = 0
    seek_status = False
    if beam_temperature_goal > 0:
        while seek_count_iter < seek_max_iter and seek_status is False:
            seek_count_iter += 1
            protection_thickness_ = np.average([seek_ubound, seek_lbound])
            inputs_steel_heat_transfer["thickness_protection"] = protection_thickness_
            t_, T_, d_ = _steel_temperature(**inputs_steel_heat_transfer)
            T_max = np.max(T_)
            y_diff_seek = T_max - beam_temperature_goal
            if abs(y_diff_seek) <= seek_tol_y:
                seek_status = True
            elif T_max > beam_temperature_goal:  # steel is too hot, increase intrumescent paint thickness
                seek_lbound = protection_thickness_
            else:  # steel is too cold, increase intrumescent paint thickness
                seek_ubound = protection_thickness_

        # BEAM FIRE RESISTANCE PERIOD IN ISO 834

        # Make steel time-temperature curve when exposed to the given ambient temperature, i.e. ISO 834.
        inputs_steel_heat_transfer["time"] = iso834_time
        inputs_steel_heat_transfer["temperature_ambient"] = iso834_temperature
        time_, temperature_steel, data_all = _steel_temperature(**inputs_steel_heat_transfer)

        # re-arrange time and steel temperature, which will be used later, to prevent interpolation boundary error. If boundaries are breached, the corresponding min or max of time will be returned, i.e. 0 and 2 hrs
        time_ = np.concatenate((np.array([0]), time_, np.array([time_[-1]])))
        temperature_steel = np.concatenate((np.array([-1]), temperature_steel, np.array([1e12])))

        # perform interpolation for teq based on acutal steel temperature and iso steel temperature
        interp_ = interp1d(temperature_steel, time_, kind="linear", bounds_error=False, fill_value=-1)
        time_fire_resistance = interp_(beam_temperature_goal)
    else:
        time_fire_resistance, T_max, protection_thickness_ = -1, -1, -1

    return time_fire_resistance, seek_status, window_open_fraction, fire_load_density, fire_spread_speed, beam_position, temperature_max_near_field, fire_type, T_max, protection_thickness_, seek_count_iter, temperature_steel_max, index


def mc_inputs_generator(dict_extra_variables_to_add=None, dir_file=str):
    steel_prop = Thermal()

    #   Handy interim functions

    def linear_distribution(min, max, prob): return ((max - min) * prob) + min

    # ------------------------------------------------------------------------------------------------------------------
    #   Define the inputs from file
    # ------------------------------------------------------------------------------------------------------------------

    # dict_setting_vars = dict()

    # Read input variables from external text file
    with open(str(dir_file), "r") as file_inputs:
        string_inputs = file_inputs.read()
    dict_vars_0 = kwargs_from_text(string_inputs)
    if dict_extra_variables_to_add:
        dict_vars_0.update(dict_extra_variables_to_add)
    dict_vars_0["beam_c"] = steel_prop.c()

    # dict_vars_0_ is used for making the returned dataframe, hence it passes necessary variables only
    dict_vars_0_ = copy.copy(dict_vars_0)
    simulations = dict_vars_0["simulations"]

    # Variable group definition
    list_setting_vars = ["simulations", "steel_temp_failure", "n_proc", "building_height", "lhs_criterion"]
    list_interim_vars = ["qfd_std", "qfd_mean", "qfd_ubound", "qfd_lbound", "glaz_min", "glaz_max", "beam_min", "beam_max", "com_eff_min", "com_eff_max", "spread_min", "spread_max", "avg_nft"]

    # Extract separated
    dict_setting_vars = {k: None for k in list_setting_vars}
    dict_dist_vars = {key: None for key in list_interim_vars}
    for key in dict_setting_vars:
        if key in dict_vars_0:
            dict_setting_vars[key] = dict_vars_0[key]
            del dict_vars_0[key]
    for key in dict_dist_vars:
        if key in dict_vars_0:
            dict_dist_vars[key] = dict_vars_0[key]
            del dict_vars_0[key]

    for key in list(dict_vars_0_.keys()):
        if key in list_setting_vars:
            del dict_vars_0_[key]
        elif key in list_interim_vars:
            del dict_vars_0_[key]

    for key in dict_vars_0_: dict_vars_0_[key] = [dict_vars_0_[key]] * simulations

    # ------------------------------------------------------------------------------------------------------------------
    # Distribution variables
    # ------------------------------------------------------------------------------------------------------------------

    # lhs_mat = lhs(n=6, samples=simulations, criterion=dict_setting_vars["lhs_criterion"])
    lhs_mat = latin_hypercube_sampling(num_samples=simulations, num_arguments=6)

    #   Set distribution mean and standard dev

    qfd_std = dict_dist_vars["qfd_std"]  # Fire load density - Gumbel distribution - standard dev [MJ/sq.m]
    qfd_mean = dict_dist_vars["qfd_mean"]  # Fire load density - Gumbel distribution - mean [MJ/sq.m]
    qfd_ubound = dict_dist_vars["qfd_ubound"]  # Fire load density - Gumbel distribution - upper limit [MJ/sq.m]
    qfd_lbound = dict_dist_vars["qfd_lbound"]  # Fire load density - Gumbel distribution - lower limit [MJ/sq.m]
    glaz_min = dict_dist_vars["glaz_min"]  # Min glazing fall-out fraction [-] - Linear dist
    glaz_max = dict_dist_vars["glaz_max"]  # Max glazing fall-out fraction [-]  - Linear dist
    beam_min = dict_dist_vars["beam_min"]  # Min beam location relative to compartment length for TFM [-]  - Linear dist
    beam_max = dict_dist_vars["beam_max"]  # Max beam location relative to compartment length for TFM [-]  - Linear dist
    com_eff_min = dict_dist_vars["com_eff_min"]  # Min combustion efficiency [-]  - Linear dist
    com_eff_max = dict_dist_vars["com_eff_max"]  # Max combustion efficiency [-]  - Linear dist
    spread_min = dict_dist_vars["spread_min"]  # Min spread rate for TFM [m/s]  - Linear dist
    spread_max = dict_dist_vars["spread_max"]  # Max spread rate for TFM [m/s]  - Linear dist
    avg_nft = dict_dist_vars["avg_nft"]  # TFM near field temperature - Norm distribution - mean [C]

    #   Calculate gumbel parameters for qfd
    qfd_scale = (qfd_std * (6 ** 0.5)) / np.pi
    qfd_loc = qfd_mean - (0.5722 * qfd_scale)
    qfd_dist = gumbel_r(loc=qfd_loc, scale=qfd_scale)
    qfd_p_l, qfd_p_u = qfd_dist.cdf(qfd_lbound), qfd_dist.cdf(qfd_ubound)
    # todo: do limits for qfd

    lhs_mat[:, 1] = latin_hypercube_sampling(num_samples=simulations, sample_min=qfd_p_l, sample_max=qfd_p_u).flatten()

    #   Near field standard deviation
    std_nft = (1.939 - (np.log(avg_nft) * 0.266)) * avg_nft

    #   Convert LHS probabilities to distribution invariants
    comb_lhs = linear_distribution(com_eff_min, com_eff_max, lhs_mat[:, 0])
    qfd_lhs = gumbel_r(loc=qfd_loc, scale=qfd_scale).ppf(lhs_mat[:, 1]) * comb_lhs
    glaz_lhs = linear_distribution(glaz_min, glaz_max, lhs_mat[:, 2])
    beam_lhs = linear_distribution(beam_min, beam_max, lhs_mat[:, 3]) * dict_vars_0["room_depth"]
    spread_lhs = linear_distribution(spread_min, spread_max, lhs_mat[:, 4])
    nft_lhs = norm(loc=avg_nft, scale=std_nft).ppf(lhs_mat[:, 5])

    nft_lhs[nft_lhs>1200] = 1200  # todo: reference?

    # ------------------------------------------------------------------------------------------------------------------
    # Create input kwargs for mc calculation
    # ------------------------------------------------------------------------------------------------------------------

    list_kwargs = []
    for i in range(0, simulations):
        if qfd_lbound > qfd_lhs[i] > qfd_ubound:  # Fire load density is outside limits
            continue
        x_ = dict_vars_0.copy()
        x_.update({"window_open_fraction": glaz_lhs[i],
                   "fire_load_density": qfd_lhs[i],
                   "fire_spread_speed": spread_lhs[i],
                   "beam_position": beam_lhs[i],
                   "temperature_max_near_field": nft_lhs[i],
                   "index": i},)
        list_kwargs.append(x_)

    dict_vars_0_["window_open_fraction"] = glaz_lhs
    dict_vars_0_["fire_load_density"] = qfd_lhs
    dict_vars_0_["fire_spread_speed"] = spread_lhs
    dict_vars_0_["beam_position"] = beam_lhs
    dict_vars_0_["temperature_max_near_field"] = nft_lhs

    df_args = df(dict_vars_0_)

    return list_kwargs, dict_setting_vars


def mc_inputs_generator2(dict_extra_variables_to_add=None, dir_file=str):
    steel_prop = Thermal()

    #   Handy interim functions

    def linear_distribution(min, max, prob): return ((max - min) * prob) + min

    # ------------------------------------------------------------------------------------------------------------------
    #   Define the inputs from file
    # ------------------------------------------------------------------------------------------------------------------

    # dict_setting_vars = dict()

    # Read input variables from external text file
    with open(str(dir_file), "r") as file_inputs:
        string_inputs = file_inputs.read()
    dict_vars_0 = kwargs_from_text(string_inputs)
    if dict_extra_variables_to_add:
        dict_vars_0.update(dict_extra_variables_to_add)
    dict_vars_0["beam_c"] = steel_prop.c()

    # dict_vars_0_ is used for making the returned dataframe, hence it passes necessary variables only
    dict_vars_0_ = copy.copy(dict_vars_0)
    simulations = dict_vars_0["simulations"]

    # Variable group definition
    list_setting_vars = ["simulations", "steel_temp_failure", "n_proc", "building_height", "select_fires_teq", "select_fires_teq_tol"]
    list_interim_vars = ["qfd_std", "qfd_mean", "qfd_ubound", "qfd_lbound", "glaz_min", "glaz_max", "beam_min", "beam_max", "com_eff_min", "com_eff_max", "spread_min", "spread_max", "avg_nft"]

    # Extract separated
    dict_setting_vars = {k: None for k in list_setting_vars}
    dict_dist_vars = {key: None for key in list_interim_vars}
    for key in dict_setting_vars:
        if key in dict_vars_0:
            dict_setting_vars[key] = dict_vars_0[key]
            del dict_vars_0[key]
    for key in dict_dist_vars:
        if key in dict_vars_0:
            dict_dist_vars[key] = dict_vars_0[key]
            del dict_vars_0[key]

    for key in list(dict_vars_0_.keys()):
        if key in list_setting_vars:
            del dict_vars_0_[key]
        elif key in list_interim_vars:
            del dict_vars_0_[key]

    for key in dict_vars_0_: dict_vars_0_[key] = [dict_vars_0_[key]] * simulations

    # ------------------------------------------------------------------------------------------------------------------
    # Distribution variables
    # ------------------------------------------------------------------------------------------------------------------

    # lhs_mat = lhs(n=6, samples=simulations, criterion=dict_setting_vars["lhs_criterion"])
    lhs_mat = latin_hypercube_sampling(num_samples=simulations, num_arguments=6)

    #   Set distribution mean and standard dev

    qfd_std = dict_dist_vars["qfd_std"]  # Fire load density - Gumbel distribution - standard dev [MJ/sq.m]
    qfd_mean = dict_dist_vars["qfd_mean"]  # Fire load density - Gumbel distribution - mean [MJ/sq.m]
    qfd_ubound = dict_dist_vars["qfd_ubound"]  # Fire load density - Gumbel distribution - upper limit [MJ/sq.m]
    qfd_lbound = dict_dist_vars["qfd_lbound"]  # Fire load density - Gumbel distribution - lower limit [MJ/sq.m]
    glaz_min = dict_dist_vars["glaz_min"]  # Min glazing fall-out fraction [-] - Linear dist
    glaz_max = dict_dist_vars["glaz_max"]  # Max glazing fall-out fraction [-]  - Linear dist
    beam_min = dict_dist_vars["beam_min"]  # Min beam location relative to compartment length for TFM [-]  - Linear dist
    beam_max = dict_dist_vars["beam_max"]  # Max beam location relative to compartment length for TFM [-]  - Linear dist
    com_eff_min = dict_dist_vars["com_eff_min"]  # Min combustion efficiency [-]  - Linear dist
    com_eff_max = dict_dist_vars["com_eff_max"]  # Max combustion efficiency [-]  - Linear dist
    spread_min = dict_dist_vars["spread_min"]  # Min spread rate for TFM [m/s]  - Linear dist
    spread_max = dict_dist_vars["spread_max"]  # Max spread rate for TFM [m/s]  - Linear dist
    avg_nft = dict_dist_vars["avg_nft"]  # TFM near field temperature - Norm distribution - mean [C]

    #   Calculate gumbel parameters for qfd
    qfd_scale = qfd_std * (6 ** 0.5) / np.pi
    qfd_loc = qfd_mean - (0.57722 * qfd_scale)
    qfd_dist = gumbel_r(loc=qfd_loc, scale=qfd_scale)
    qfd_p_l, qfd_p_u = qfd_dist.cdf(qfd_lbound), qfd_dist.cdf(qfd_ubound)
    # todo: do limits for qfd

    lhs_mat[:, 1] = latin_hypercube_sampling(num_samples=simulations, sample_min=qfd_p_l, sample_max=qfd_p_u).flatten()

    #   Near field standard deviation
    std_nft = (1.939 - (np.log(avg_nft) * 0.266)) * avg_nft

    #   Convert LHS probabilities to distribution invariants
    comb_lhs = linear_distribution(com_eff_min, com_eff_max, lhs_mat[:, 0])
    qfd_lhs = gumbel_r(loc=qfd_loc, scale=qfd_scale).ppf(lhs_mat[:, 1]) * comb_lhs
    glaz_lhs = linear_distribution(glaz_min, glaz_max, lhs_mat[:, 2])
    beam_lhs = linear_distribution(beam_min, beam_max, lhs_mat[:, 3]) * dict_vars_0["room_depth"]
    spread_lhs = linear_distribution(spread_min, spread_max, lhs_mat[:, 4])
    nft_lhs = norm(loc=avg_nft, scale=std_nft).ppf(lhs_mat[:, 5])

    nft_lhs[nft_lhs>1200] = 1200  # todo: reference?

    # ------------------------------------------------------------------------------------------------------------------
    # Create input kwargs for mc calculation
    # ------------------------------------------------------------------------------------------------------------------

    list_kwargs = []
    for i in range(0, simulations):
        if qfd_lbound > qfd_lhs[i] > qfd_ubound:  # Fire load density is outside limits
            continue
        x_ = dict_vars_0.copy()
        x_.update({"window_open_fraction": glaz_lhs[i],
                   "fire_load_density": qfd_lhs[i],
                   "fire_spread_speed": spread_lhs[i],
                   "beam_position": beam_lhs[i],
                   "temperature_max_near_field": nft_lhs[i],
                   "index": i},)
        list_kwargs.append(x_)

    dict_vars_0_["window_open_fraction"] = glaz_lhs
    dict_vars_0_["fire_load_density"] = qfd_lhs
    dict_vars_0_["fire_spread_speed"] = spread_lhs
    dict_vars_0_["beam_position"] = beam_lhs
    dict_vars_0_["temperature_max_near_field"] = nft_lhs
    dict_vars_0_["index"] = np.arange(0, simulations, 1, int)

    df_args = df(dict_vars_0_)
    df_args.set_index("index", inplace=True)

    return df_args, dict_setting_vars


def mc_post_processing(x, x_find=None, y_find=None):
    # work out x_sorted, y
    x_raw = np.sort(x)
    y_raw = np.arange(1, len(x_raw) + 1) / len(x_raw)

    cdf_x = interp1d(x_raw, y_raw)
    cdf_y = interp1d(y_raw, x_raw)

    # work out pdf
    pdf_x = stats.gaussian_kde(x_raw, bw_method="scott")
    x_f = np.linspace(x_raw.min() - 1, x_raw.max() + 1, 2000)

    y_pdf = pdf_x.evaluate(x_f)
    y_cdf = np.cumsum(y_pdf) * ((x_raw.max()-x_raw.min()+2) / 2000)

    # find y according x_find and/ or x according y_find
    xy_found = []
    if x_find is not None:
        y_found = cdf_x(x_find)
        xy_found.append(*list(zip(x_find, y_found)))
    if y_find is not None:
        x_found = cdf_y(y_find)
        xy_found.append(*list(zip(x_found, y_find)))

    xy_found = np.asarray(xy_found)

    return x_raw, y_raw, x_f, y_cdf, xy_found
