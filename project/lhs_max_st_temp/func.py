# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats.distributions import norm, gumbel_r
from pyDOE import lhs
from project.dat.steel_carbon import Thermal
import project.lhs_max_st_temp.ec_param_fire as pf
import project.lhs_max_st_temp.tfm_alt as _fire_travelling
from project.func.temperature_steel_section import protected_steel_eurocode as _steel_temperature
import logging
from project.func.temperature_fires import standard_fire_iso834
from scipy.interpolate import interp1d
from project.func.temperature_fires import parametric_eurocode1 as _fire_param
from project.func.kwargs_from_text import kwargs_from_text
from scipy import stats
from project.func.tools_number import distribute_numbers_cartesian_product


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
            window_open_fraction
        )
        temps += 273.15
        fire_type = 1

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
        "rho_steel": beam_rho,
        "c_steel_T": beam_c,
        "area_steel_section": beam_cross_section_area,
        "k_protection": protection_k,
        "rho_protection": protection_rho,
        "c_protection": protection_c,
        "thickness_protection": protection_thickness,
        "perimeter_protected": protection_protected_perimeter,
    }
    max_temp = np.max(_steel_temperature(**inputs_steel_heat_transfer)[1])

    return max_temp, window_open_fraction, fire_load_density, fire_spread_speed, beam_position, temperature_max_near_field, fire_type


def mc_fr_calculation(
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
            window_open_fraction
        )
        temps += 273.15
        fire_type = 1

    # print("LHS_model_realisation_count =", i + 1, "Of", lhs_iterations)

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
                                  "is_terminate_peak_steel_temperature": True}
    # max_temp = np.max(_steel_temperature(**inputs_steel_heat_transfer)[1])

    # MATCH PEAK STEEL TEMPERATURE BY ADJUSTING PROTECTION LAYER THICKNESS
    # todo: if seeking unsuccessful?
    seek_max_iter = 20  # todo: move to input
    seek_ubound = 0.1  # todo: move to input
    seek_lbound = 0.001  # todo: move to input
    seet_tol_y = 1  # todo: move to inputs
    seek_count_iter = 0
    seek_status = False
    while seek_count_iter < seek_max_iter and seek_status is False:
        seek_count_iter += 1
        protection_thickness_ = np.average([seek_ubound, seek_lbound])
        inputs_steel_heat_transfer["thickness_protection"] = protection_thickness_
        t_, T_, d_ = _steel_temperature(**inputs_steel_heat_transfer)
        T_max = np.max(T_)
        y_diff_seek = T_max - beam_temperature_goal
        if abs(y_diff_seek) <= seet_tol_y:
            seek_status = True
        elif T_max > beam_temperature_goal:  # steel is too hot, increase intrumescent paint thickness
            seek_lbound = protection_thickness_
        else:  # steel is too cold, increase intrumescent paint thickness
            seek_ubound = protection_thickness_

    # BEAM FIRE RESISTANCE PERIOD IN ISO 834
    # Make steel time-temperature curve in given fire (i.e. ISO 834)
    inputs_steel_heat_transfer["time"] = iso834_time
    inputs_steel_heat_transfer["temperature_ambient"] = iso834_temperature
    time_, temperature_steel, data_all = _steel_temperature(**inputs_steel_heat_transfer)
    interp_ = interp1d(temperature_steel, time_, kind="linear", bounds_error=False, fill_value=-1)
    time_fire_resistance = interp_(beam_temperature_goal)

    return time_fire_resistance, seek_status, window_open_fraction, fire_load_density, fire_spread_speed, beam_position, temperature_max_near_field, fire_type, T_max, protection_thickness_, seek_count_iter


def mc_inputs_generator(simulation_count, dict_extra_inputs=None, dir_file="inputs.txt"):
    steel_prop = Thermal()

    #   Handy interim functions

    def linear_distribution(min, max, prob): return ((max - min) * prob) + min

    #   Define the inputs

    lhs_iterations = simulation_count

    x = dict()

    # Read input variables from external text file
    with open(dir_file, "r") as file_inputs:
        string_inputs = file_inputs.read()
    dict_inputs = kwargs_from_text(string_inputs)
    x.update(dict_inputs)

    # Physical properties

    x["beam_c"] = steel_prop.c()

    if dict_extra_inputs is not None:
        x.update(dict_extra_inputs)

    #   Set distribution mean and standard dev
    qfd_std = x["dist_s_fire_load_density"]  # Fire load density - Gumbel distribution - standard dev [MJ/sq.m]
    qfd_mean = x["dist_m_fire_load_density"]  # Fire load density - Gumbel distribution - mean [MJ/sq.m]
    qfd_ubound = x["dist_u_fire_load_density"]  # Fire load density - Gumbel distribution - upper limit [MJ/sq.m]
    qfd_lbound = x["dist_l_fire_load_density"]  # Fire load density - Gumbel distribution - lower limit [MJ/sq.m]
    glaz_min = x["dist_l_glazing_break_percentage"]  # Min glazing fall-out fraction [-] - Linear dist
    glaz_max = x["dist_u_glazing_break_percentage"]  # Max glazing fall-out fraction [-]  - Linear dist
    beam_min = x["dist_l_beam_location"]  # Min beam location relative to compartment length for TFM [-]  - Linear dist
    beam_max = x["dist_u_beam_location"]  # Max beam location relative to compartment length for TFM [-]  - Linear dist
    com_eff_min = x["dist_l_combustion_efficiency"]  # Min combustion efficiency [-]  - Linear dist
    com_eff_max = x["dist_u_combustion_effeciency"]  # Max combustion efficiency [-]  - Linear dist
    spread_min = x["dist_l_fire_spread"]  # Min spread rate for TFM [m/s]  - Linear dist
    spread_max = x["dist_u_fire_spread"]  # Max spread rate for TFM [m/s]  - Linear dist
    avg_nft = x["dist_m_near_field_temperature"]  # TFM near field temperature - Norm distribution - mean [C]

    #   Create random number array for each stochastic variable
    def random_numbers_lhs(n=lhs_iterations, l_lim=0, u_lim=1):
        return lhs(1, samples=n).flatten() * (u_lim - l_lim) + l_lim

    n = [int(lhs_iterations**(1/6))+1]
    dist_numbers = distribute_numbers_cartesian_product(n*6)

    #   Calculate gumbel parameters
    qfd_scale = (qfd_std * (6 ** 0.5)) / np.pi
    qfd_loc = qfd_mean - (0.5722 * qfd_scale)

    #   Near field standard deviation
    std_nft = (1.939 - (np.log(avg_nft) * 0.266)) * avg_nft

    #   Convert LHS probabilities to distribution invariants
    # comb_lhs = linear_distribution(com_eff_min, com_eff_max, random_numbers_lhs())
    # qfd_dist = gumbel_r(loc=qfd_loc, scale=qfd_scale)
    # qfd_p_l, qfd_p_u = qfd_dist.cdf(qfd_lbound), qfd_dist.cdf(qfd_ubound)
    # qfd_lhs = gumbel_r(loc=qfd_loc, scale=qfd_scale).ppf(random_numbers_lhs(l_lim=qfd_p_l, u_lim=qfd_p_u)) * comb_lhs
    # glaz_lhs = linear_distribution(glaz_min, glaz_max, random_numbers_lhs())
    # beam_lhs = linear_distribution(beam_min, beam_max, random_numbers_lhs()) * x["room_depth"]
    # spread_lhs = linear_distribution(spread_min, spread_max, random_numbers_lhs())
    # nft_lhs = norm(loc=avg_nft, scale=std_nft).ppf(random_numbers_lhs())

    comb_lhs = linear_distribution(com_eff_min, com_eff_max, dist_numbers[:,0])
    qfd_dist = gumbel_r(loc=qfd_loc, scale=qfd_scale)
    qfd_p_l, qfd_p_u = qfd_dist.cdf(qfd_lbound), qfd_dist.cdf(qfd_ubound)
    qfd_lhs = gumbel_r(loc=qfd_loc, scale=qfd_scale).ppf(dist_numbers[:,1]) * comb_lhs
    glaz_lhs = linear_distribution(glaz_min, glaz_max, dist_numbers[:,2])
    beam_lhs = linear_distribution(beam_min, beam_max, dist_numbers[:,3]) * x["room_depth"]
    spread_lhs = linear_distribution(spread_min, spread_max, dist_numbers[:,4])
    nft_lhs = norm(loc=avg_nft, scale=std_nft).ppf(dist_numbers[:,5])

    # delete these items as they are nolonger used and will cause error if passed on
    del x["dist_s_fire_load_density"]  # Fire load density - Gumbel distribution - standard dev [MJ/sq.m]
    del x["dist_m_fire_load_density"]  # Fire load density - Gumbel distribution - mean [MJ/sq.m]
    del x["dist_u_fire_load_density"]  # Fire load density - Gumbel distribution - upper limit [MJ/sq.m]
    del x["dist_l_fire_load_density"]  # Fire load density - Gumbel distribution - lower limit [MJ/sq.m]
    del x["dist_l_glazing_break_percentage"]  # Min glazing fall-out fraction [-] - Linear dist
    del x["dist_u_glazing_break_percentage"]  # Max glazing fall-out fraction [-]  - Linear dist
    del x["dist_l_beam_location"]  # Min beam location relative to compartment length for TFM [-]  - Linear dist
    del x["dist_u_beam_location"]  # Max beam location relative to compartment length for TFM [-]  - Linear dist
    del x["dist_l_combustion_efficiency"]  # Min combustion efficiency [-]  - Linear dist
    del x["dist_u_combustion_effeciency"]  # Max combustion efficiency [-]  - Linear dist
    del x["dist_l_fire_spread"]  # Min spread rate for TFM [m/s]  - Linear dist
    del x["dist_u_fire_spread"]  # Max spread rate for TFM [m/s]  - Linear dist
    del x["dist_m_near_field_temperature"]  # TFM near field temperature - Norm distribution - mean [C]

    list_inputs = []
    for i in range(0, lhs_iterations):
        if qfd_lbound > qfd_lhs[i] > qfd_ubound:  # Fire load density is outside limits
            continue
        x_ = x.copy()
        x_.update({"window_open_fraction": glaz_lhs[i],
                   "fire_load_density": qfd_lhs[i],
                   "fire_spread_speed": spread_lhs[i],
                   "beam_position": beam_lhs[i],
                   "temperature_max_near_field": min([nft_lhs[i], 1200]),
                   "index": i},)
        list_inputs.append(x_)

    return list_inputs


def mc_post_processing(x, x_find=None, y_find=None):
    # work out x_sorted, y
    x_raw = np.sort(x)
    y_raw = np.arange(1, len(x_raw) + 1) / len(x_raw)

    cdf_x = interp1d(x_raw, y_raw)
    cdf_y = interp1d(y_raw, x_raw)

    # work out pdf
    pdf_x = stats.gaussian_kde(x_raw, bw_method="scott")
    # x_f = np.arange(x_raw.min() - 1, x_raw.max() + 1)
    x_f = np.linspace(x_raw.min() - 1, x_raw.max() + 1, 2000)

    y_pdf = pdf_x.evaluate(x_f)
    y_cdf = np.cumsum(y_pdf) * ((x_raw.max()-x_raw.min()+2) / 2000)
    # cdf_x = interp1d(x_f, y_cdf)  # TO BE CONFIRMED WHETHER THESE TO BE USED.
    # cdf_y = interp1d(y_cdf, x_f)

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
