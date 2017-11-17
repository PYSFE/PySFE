# -*- coding: utf-8 -*-
import numpy as np
import copy

# def parametric_eurocode1(
#         total_enclosure_area,
#         floor_area,
#         opening_area,
#         opening_height,
#         density_boundary,
#         specific_heat_boundary,
#         thermal_conductivity_boundary,
#         fire_load_density_floor,
#         fire_growth_rate,
#         time_step=0.1,
#         time_extend=1800
# ):
#     """DESCRIPTION:
#     Generates the Eurocode parametric fire curve defined in [BS EN 1991-1-2:2002, Annex A]
#     NOTE:
#         *   When estimating extinction time, a upper bound value of 12 hours is used for approximation. The current
#             version is not capable estimating extinction time if it happened greater than 12 hrs. In addition, 1000
#             loops are set to maximum for goal seek.
#         *   0.02 <= opening factor <= 0.002
#         *   100 <= b (thermal inertia) <= 2200
#     :param total_enclosure_area:            [float][m²]     Total internal surface area, including openings.
#     :param floor_area:                      [float][m²]     Floor area.
#     :param opening_area:                    [float][m²]     Vertical open area.
#     :param opening_height:                  [float][m]      Weighted average vertical openings height.
#     :param density_boundary:                [float][kg/m³]  Density of enclosure boundary.
#     :param specific_heat_boundary:          [float][J/kg/K] Specific heat of enclosure.
#     :param thermal_conductivity_boundary:   [float][W/m/K]  Thermal conductivity of enclosure.
#     :param fire_load_density_floor:         [float][J/m²]   Fuel load on floor area.
#     :param fire_growth_rate:                [string][-]     Can be either "low", "medium" or "fast".
#     :param time_step:                       [float][s]      Time step between each interval.
#     :param time_extend:                     [float][s]      Time extended after fire extinguished.
#     :return time:                           [ndarray][K]    Time array incorporating with temperatures.
#     :return temperature:                    [ndarray][K]    Temperature array incorporating with time.
#     """
#
#     # HELPER FUNCTIONS
#
#     def _temperature_heating_phase(time_array_heating, time_peak_temperature, time_limiting, lambda_, lambda_limiting):
#         if time_peak_temperature == time_limiting:
#             # (A.2b)
#             tt = time_array_heating * lambda_limiting
#         else:
#             # (A.2a)
#             tt = time_array_heating * lambda_
#
#         return 20 + 1325 * (1 - 0.324 * np.exp(-0.2*tt) - 0.204 * np.exp(-1.7*tt) - 0.472 * np.exp(-19*tt))
#
#     def _time_extinction(fire_load_density_total, opening_factor, lambda_, time_peak_temperature, time_limiting, temperature_peak):
#         tt_max = (
#                      0.2e-3 * fire_load_density_total / opening_factor) * lambda_  # (A.12) explicitly for cooling phase
#         if (time_peak_temperature > time_limiting):  # (A.12), explicitly for cooling phase
#             x = 1.0
#         elif (time_peak_temperature == time_limiting):
#             x = time_limiting * lambda_ / tt_max
#         gas_temperature = temperature_peak
#
#         t_ubound = 12 * 3600
#         t_lbound = time_peak_temperature
#         t = None
#
#         count_loops = 0
#         while not 19.5 < gas_temperature < 20.5 and count_loops <= 10000:
#             if t is None:
#                 t = 0.5 * (t_ubound + t_lbound)
#             else:
#                 if gas_temperature > 20:
#                     t_lbound = t
#                     t = 0.5 * (t_lbound + t_ubound)
#                 elif gas_temperature < 20:
#                     t_ubound = t
#                     t = 0.5 * (t_lbound + t_ubound)
#             if tt_max <= 0.5: # (A.11a)
#                 gas_temperature = temperature_peak - 625 * (t * lambda_ - tt_max * x)
#             elif 0.5 < tt_max < 2: # (A.11a)
#                 gas_temperature = temperature_peak - 250 * (3 - tt_max) * (t * lambda_ - tt_max * x)
#             elif tt_max >= 2: # (A.11a)
#                 gas_temperature = temperature_peak - 250 * (t * lambda_ - tt_max * x)
#             count_loops += 1
#
#         return t
#
#     def _temperature_cooling_phase(time_array_cooling, time_peak_temperature, time_limiting, lambda_, fire_load_density_total, opening_factor, temperature_peak):
#         tt = time_array_cooling * lambda_  # (A.12), explicitly for cooling phase
#         ttMax = (0.2e-3 * fire_load_density_total / opening_factor) * lambda_ # (A.12) explicitly for cooling phase
#         if time_peak_temperature > time_limiting:  # (A.12), explicitly for cooling phase
#             x = 1.0
#         elif time_peak_temperature == time_limiting:
#             x = time_limiting * lambda_ / ttMax
#         else:
#             x = None
#
#         if ttMax <= 0.5:  # (A.11a)
#             gas_temperature = temperature_peak - 625 * (tt - ttMax * x)
#         elif 0.5 < ttMax < 2:  # (A.11a)
#             gas_temperature = temperature_peak - 250 * (3 - ttMax) * (tt - ttMax * x)
#         elif ttMax >= 2:  # (A.11a)
#             gas_temperature = temperature_peak - 250 * (tt - ttMax * x)
#         else:
#             gas_temperature = np.zeros(np.shape(tt))
#
#         return gas_temperature
#
#     # UNITS CONVERSION TO FIT EQUATIONS
#     time_step /= 3600.  # seconds -> hours
#     time_extend /= 3600.  # seconds -> hours
#     fire_load_density_floor /= 1e6  # [J/m2] -> [MJ/m2]
#     fire_growth_rate /= 20
#
#     # DERIVED PROPERTIES
#
#     # [BS EN 1991-1-2:2002, Annex A, (10)]
#     dict_fire_growth = {"slow": 25.0 / 60.0, "medium": 20.0 / 60.0, "fast": 15.0 / 60.0}
#
#     # opening_factor = opening_area * np.sqrt(opening_height) / total_enclosure_area
#     opening_factor = 0.2
#
#     # b = np.sqrt(density_boundary * specific_heat_boundary * thermal_conductivity_boundary)
#     b = 1500
#
#     lambda_ = np.power(opening_factor / b, 2) / np.power(0.04 / 1160, 2)
#
#     fire_load_density_total = fire_load_density_floor * (floor_area / total_enclosure_area)
#
#     # [BS EN 1991-1-2:2002, Annex A, (10)]
#     if isinstance(fire_growth_rate, str):
#         time_limiting = dict_fire_growth[fire_growth_rate]
#     else:
#         time_limiting = fire_growth_rate
#
#     # [BS EN 1991-1-2:2002, Annex A, A.9]
#     opening_factor_limiting = 0.1e-3 * fire_load_density_total / time_limiting
#
#     lambda_limiting = np.power((opening_factor_limiting / b), 2) / np.power(0.04 / 1160, 2)
#
#     if opening_factor > 0.04 and fire_load_density_total < 75 and b < 1160:
#         # (A.10)
#         lambda_limiting *= \
#             (1 + ((opening_factor - 0.04) / 0.04) * ((fire_load_density_total - 75) / 75) * ((1160 - b) / 1160))
#
#     time_peak_temperature = max([0.2e-3 * fire_load_density_total / opening_factor, time_limiting])
#
#     # CALCULATE TIME & TEMPERATURE FOR HEATING PHASE
#     time_array_heating = np.arange(0, int(time_peak_temperature/time_step)*time_step+time_step, time_step)
#     temperature_array_heating = _temperature_heating_phase(time_array_heating, time_peak_temperature, time_limiting, lambda_, lambda_limiting)
#     temperature_peak = max(temperature_array_heating)
#
#     # CALCULATE TIME & TEMPERATURE FOR COOLING PHASE
#     extinguish_time = _time_extinction(fire_load_density_total, opening_factor, lambda_, time_peak_temperature, time_limiting, temperature_peak)
#     time_array_cooling = np.arange(time_array_heating.max()+time_step, int(extinguish_time/time_step)*time_step+time_step, time_step)
#     temperature_array_cooling = _temperature_cooling_phase(time_array_cooling, time_peak_temperature, time_limiting, lambda_, fire_load_density_total, opening_factor, temperature_peak)
#     temperature_array_cooling[temperature_array_cooling < 25] = 25
#
#     # EXTEND PHASE
#     time_array_extend = np.arange(time_array_cooling.max() + time_step, time_array_cooling.max() + time_step + time_extend, time_step)
#     temperature_array_extend = np.full((np.size(time_array_extend),), 25)
#
#     # Assemble array
#     time = np.concatenate((time_array_heating,time_array_cooling,time_array_extend))
#     temperature = np.concatenate((temperature_array_heating, temperature_array_cooling, temperature_array_extend))
#
#     # Convert time to seconds
#     time *= 3600
#     temperature += 273.15
#
#     return time, temperature

import logging
logging.basicConfig(filename="log.txt", level=logging.DEBUG)
log = logging.getLogger(__name__)


def parametric_eurocode1(A_t, A_f, A_v, h_eq, q_fd, lambda_, rho, c, t_lim, time_end=7200, time_step=1, time_start=0, time_padding = (0, 0),temperature_initial=293.15):
    """Function Description: (SI UNITS ONLY)
    This function calculates the time-temperature curve according to Eurocode 1 part 1-2, Appendix A.
    :param A_t:
    :param A_f:
    :param A_v:
    :param h_eq:
    :param q_fd:
    :param lambda_:
    :param rho:
    :param c:
    :param t_lim:
    :param time_end:
    :param time_step:
    :param time_start:
    :param time_padding:
    :param temperature_initial:
    :return t:
    :return T_g:
    """
    # Reference: Eurocode 1991-1-2; Jean-Marc Franssen, Paulo Vila Real (2010) - Fire Design of Steel Structures

    # UNITS: SI -> Equations
    q_fd /= 1e6  # [J/m2] -> [MJ/m2]
    t_lim /= 3600  # [s] -> [hr]
    time_end /= 3600  # [s] -> [hr]
    time_step /= 3600  # [s] -> [hr]
    time_start /= 3600  # [s] -> [hr]
    temperature_initial -= 273.15  # [K] -> [C]

    # ACQUIRING REQUIRED VARIABLES
    t = np.arange(time_start, time_end, time_step, dtype=float)

    b = (lambda_ * rho * c) ** 0.5
    O = A_v * h_eq**0.5 / A_t
    q_td = q_fd * A_f / A_t
    Gamma = ((O/0.04)/(b/1160))**2

    t_max = 0.0002*q_td/O

    # check criteria
    if not 50 <= q_td <= 1000: log.warning("q_td = {:4.1f} not in range [50, 1000]".format(q_td))

    # CALCULATION
    def _temperature_heating(t_star, temperature_initial):
        # eq. 3.12
        T_g = 1325 * (1 - 0.324*np.exp(-0.2*t_star) - 0.204*np.exp(-1.7*t_star) - 0.472*np.exp(-19*t_star))
        T_g += temperature_initial
        return T_g

    def _temperature_cooling_vent(t_star_max, T_max, t_star):  # ventilation controlled
        # eq. 3.16
        if t_star_max <= 0.5:
            T_g = T_max - 625 * (t_star - t_star_max)
        elif 0.5 < t_star_max < 2.0:
            T_g = T_max - 250 * (3 - t_star_max) * (t_star - t_star_max)
        elif 2.0 <= t_star_max:
            T_g = T_max - 250 * (t_star - t_star_max)
        else: T_g = np.nan
        return T_g

    def _temperature_cooling_fuel(t_star_max, T_max, t_star, Gamma, t_lim):  # fuel controlled
        # eq. 3.22
        if t_star_max <= 0.5:
            T_g = T_max - 625 * (t_star - Gamma * t_lim)
        elif 0.5 < t_star_max < 2.0:
            T_g = T_max - 250 * (3 - t_star_max) * (t_star - Gamma * t_lim)
        elif 2.0 <= t_star_max:
            T_g = T_max - 250 * (t_star - Gamma * t_lim)
        else: T_g = np.nan
        return T_g

    def _variables(t, Gamma, t_max):
        t_star = Gamma * t
        t_star_max = Gamma * t_max
        return t_star, t_star_max

    def _variables_2(t, t_lim, q_td, b, O):
        O_lim = 0.0001 * q_td / t_lim
        Gamma_lim = ((O_lim/0.04)/(b/1160))**2

        if O > 0.04 and q_td < 75 and b < 1160:
            k = 1 + ((O-0.04)/(0.04)) * ((q_td-75)/(75)) * ((1160-b)/(1160))
            Gamma_lim *= k

        t_star_ = Gamma_lim * t
        t_star_max_ = Gamma_lim * t_lim
        return t_star_, t_star_max_

    t_star, t_star_max = _variables(t, Gamma, t_max)

    if t_max >= t_lim:  # ventilation controlled fire
        T_max = _temperature_heating(t_star_max, temperature_initial)
        T_heating_g = _temperature_heating(Gamma * t, temperature_initial)
        T_cooling_g = _temperature_cooling_vent(t_star_max, T_max, t_star)
        fire_type = "ventilation controlled"
    else:  # fuel controlled fire
        t_star_, t_star_max_ = _variables_2(t, t_lim, q_td, b, O)
        T_max = _temperature_heating(t_star_max_, temperature_initial)
        T_heating_g = _temperature_heating(t_star_, temperature_initial)
        T_cooling_g = _temperature_cooling_fuel(t_star_max, T_max, t_star, Gamma, t_lim)
        fire_type = "fuel controlled"

    T_g = np.minimum(T_heating_g, T_cooling_g)
    T_g[T_g<0] = 0

    data_all = {"fire_type": fire_type}

    # UNITS: Eq. -> SI
    t *= 3600
    T_g += 273.15

    return t, T_g


def standard_fire_iso834(
        time,
        temperature_initial
):
    # INPUTS CHECK
    time = np.array(time, dtype=float)
    time[time < 0] = np.nan

    # SI UNITS -> EQUATION UNITS
    temperature_initial -= 273.15  # [K] -> [C]
    time /= 60.  # [s] - [min]

    # CALCULATE TEMPERATURE BASED ON GIVEN TIME
    temperature = 345. * np.log10(time * 8. + 1.) + temperature_initial
    temperature[temperature == np.nan] = temperature_initial

    # EQUATION UNITS -> SI UNITS
    time *= 60.  # [min] -> [s]
    temperature += 273.15  # [C] -> [K]

    return time, temperature


def standard_fire_astm_e119(
        time,
        temperature_ambient
):
    time /= 1200.  # convert from seconds to hours
    temperature_ambient -= 273.15  # convert temperature from kelvin to celcius
    temperature = 750 * (1 - np.exp(-3.79553 * np.sqrt(time))) + 170.41 * np.sqrt(time) + temperature_ambient
    return temperature + 273.15  # convert from celsius to kelvin (SI unit)


def hydrocarbon_eurocode(
        time,
        temperature_initial
):
    time /= 1200.  # convert time unit from second to hour
    temperature_initial -= 273.15  # convert temperature from kelvin to celsius
    temperature = 1080 * (1 - 0.325 * np.exp(-0.167 * time) - 0.675 * np.exp(-2.5 * time)) + temperature_initial
    return temperature + 273.15


def external_fire_eurocode(
        time,
        temperature_initial
):
    time /= 1200.  # convert time from seconds to hours
    temperature_initial -= 273.15  # convert ambient temperature from kelvin to celsius
    temperature = 660 * (1 - 0.687 * np.exp(-0.32 * time) - 0.313 * np.exp(-3.8 * time)) + temperature_initial
    return temperature + 273.15  # convert temperature from celsius to kelvin


def travelling_fire(
        T_0,
        q_fd,
        RHRf,
        l,
        w,
        s,
        # A_v,
        # h_eq,
        h_s,
        l_s,
        time_ubound=10800,
        time_step=1):
    """
    :param T_0: [float][K] Initial temperature.
    :param q_fd: [float][J m2] Fire load density.
    :param RHRf: [float][W m2] Heat release rate density
    :param l: [float][m] Compartment length
    :param w: [float][m] Compartment width
    :param s: [float][m/s] Fire spread speed
    # :param A_v: [float][m2] Ventilation area
    # :param h_eq: [float][m] Weighted ventilation height
    :param h_s: [float][m] Vertical distance between element to fuel bed.
    :param l_s: [float][m] Horizontal distance between element to fire front.
    :param time_ubound: [float][s] Maximum time for the curve.
    :param time_step: [float][s] Static time step.
    :return time: [ndarray][s] An array representing time incorporating 'temperature'.
    :return temperature: [ndarray][K] An array representing temperature incorporating 'time'.
    """

    # SETTINGS
    time_lbound = 0

    # UNIT CONVERSION TO FIT EQUATIONS
    T_0 -= 273.15
    q_fd /= 1e6
    RHRf /= 1e6

    # MAKE TIME ARRAY
    time = np.arange(time_lbound, time_ubound+time_step, time_step)

    # fire_load_density_MJm2=900
    # heat_release_rate_density_MWm2=0.15
    # length_compartment_m=150
    # width_compartment_m=17.4
    # fire_spread_rate_ms=0.012
    # area_ventilation_m2=190
    # height_ventilation_opening_m=3.3
    # height_fuel_to_element_m=3.5
    # length_element_to_fire_origin_m=105

    # workout burning time etc.
    t_burn = max([q_fd / RHRf, 900.])
    t_decay = max([t_burn, l / s])
    t_lim = min([t_burn, l / s])

    # reduce resolution to fit time step for t_burn, t_decay, t_lim
    t_decay_ = round(t_decay/time_step, 0) * time_step
    t_lim_ = round(t_lim/time_step, 0) * time_step
    if t_decay_ == t_lim_: t_lim_ -= time_step

    # workout the heat release rate ARRAY (corrected with time)
    Q_growth = (RHRf * w * s * time) * (time < t_lim_)
    Q_peak = min([RHRf * w * s * t_burn, RHRf * w * l]) * (time >= t_lim_) * (time <= t_decay_)
    Q_decay = (max(Q_peak) - (time-t_decay_) * w * s * RHRf) * (time > t_decay_)
    Q_decay[Q_decay < 0] = 0
    Q = (Q_growth + Q_peak + Q_decay) * 1000.

    # workout the distance between fire_curve midian to the structural element r
    l_fire_front = s * time
    l_fire_front[l_fire_front < 0] = 0.
    l_fire_front[l_fire_front > l] = l
    l_fire_end = s * (time - t_lim)
    l_fire_end[l_fire_end < 0] = 0.
    l_fire_end[l_fire_end > l] = l
    l_fire_median = (l_fire_front + l_fire_end) / 2.
    r = np.absolute(l_s - l_fire_median)
    r[r == 0] = 0.001  # will cause crash if r = 0

    # workout the far field temperature of gas T_g
    T_g1 = (5.38 * np.power(Q / r, 2 / 3) / h_s) * ((r/h_s) > 0.18)
    T_g2 = (16.9 * np.power(Q, 2 / 3) / np.power(h_s, 5/3)) * ((r/h_s) <= 0.18)
    T_g = T_g1 + T_g2 + T_0

    T_g[T_g >= 1200.] = 1200.

    # UNIT CONVERSION TO FIT OUTPUT (SI)
    T_g += 273.15  # C -> K
    Q *= 10e6  # MJ -> J

    temperature = T_g

    data_trivial = {
        "heat release [J]": Q,
        "distance fire to element [m]": r
    }
    return time, temperature, data_trivial
