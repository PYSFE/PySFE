# -*- coding: utf-8 -*-
import numpy as np
from project.ExtFS.Configuration_factor import config_rectangle

def travelling_fire(
        fire_load_density_MJm2,
        heat_release_rate_density_MWm2,
        length_compartment_m,
        width_compartment_m,
        fire_spread_rate_ms,
        height_fuel_to_element_m,
        length_element_to_fire_origin_m,
        time_start_s,
        time_end_s,
        time_interval_s,
        nft_max_C,
        win_width_m,
        win_height_m,
        open_fract
    ):
    # make time array
    t = np.arange(time_start_s, time_end_s, time_interval_s)

    # re-assign variable names for equation readability
    q_fd = fire_load_density_MJm2
    RHRf = heat_release_rate_density_MWm2
    l = max([length_compartment_m, width_compartment_m])
    w = min([length_compartment_m, width_compartment_m])
    s = fire_spread_rate_ms
    h_s = height_fuel_to_element_m
    l_s = length_element_to_fire_origin_m

    # work out ventilation conditions
    a_v = win_height_m * win_width_m * open_fract
    Qv = 1.75 * a_v * np.sqrt(win_height_m)

    # workout burning time etc.
    t_burn = max([q_fd / RHRf, 900.])
    t_decay = max([t_burn, l / s])
    t_lim = min([t_burn, l / s])

    # reduce resolution to fit time step for t_burn, t_decay, t_lim
    t_decay_ = round(t_decay / time_interval_s, 0) * time_interval_s
    t_lim_ = round(t_lim / time_interval_s, 0) * time_interval_s
    if t_decay_ == t_lim_: t_lim_ -= time_interval_s

    # workout the heat release rate ARRAY (corrected with time)
    Q_growth = (RHRf * w * s * t) * (t < t_lim_)
    Q_peak = min([RHRf * w * s * t_burn, RHRf * w * l]) * (t >= t_lim_) * (t <= t_decay_)
    Q_decay = (max(Q_peak) - (t - t_decay_) * w * s * RHRf) * (t > t_decay_)
    Q_decay[Q_decay < 0] = 0
    Q = (Q_growth + Q_peak + Q_decay) * 1000.

    # workout the distance between fire median to the structural element r
    l_fire_front = s * t
    l_fire_front[l_fire_front < 0] = 0.
    l_fire_front[l_fire_front > l] = l
    l_fire_end = s * (t - t_lim)
    l_fire_end[l_fire_end < 0] = 0.
    l_fire_end[l_fire_end > l] = l
    l_fire_median = (l_fire_front + l_fire_end) / 2.
    r = np.absolute(l_s - l_fire_median)
    nf_width = l_fire_front - l_fire_end

    # workout the far field temperature of gas T_g

    T_g1 = (5.38 * np.power(Q / r, 2 / 3) / h_s) * (r / h_s > 0.18).astype(int)
    T_g2 = (16.9 * np.power(Q, 2 / 3) / np.power(h_s, 5/3)) * (r/h_s <= 0.18).astype(int)
    T_g = T_g1 + T_g2 + 20.
    T_g[T_g>=nft_max_C] = nft_max_C

    return t, T_g, Q, r, nf_width


if __name__ == '__main__':

    c, T_g, Q, r, fs = travelling_fire(
        fire_load_density_MJm2=900,
        heat_release_rate_density_MWm2=0.15,
        length_compartment_m=150,
        width_compartment_m=17.4,
        fire_spread_rate_ms=0.012,
        height_fuel_to_element_m=3.5,
        length_element_to_fire_origin_m=105,
        time_start_s=0,
        time_end_s=22080,
        time_interval_s=60,
        nft_max_C=1200,
        win_width_m=0,
        win_height_m=0,
        open_fract=1
    )

    room_height = 3.5
    width = 1
    room_height = 5
    receiver_distance = 2

    c = np.linspace(0, 10, 1000)
    r = np.abs(np.linspace(-5, 5, 1000))
    fs = np.ones_like(r) * width

    phi = np.zeros_like(c, dtype=float)
    for i, v in enumerate(c):
        fire_width2 = fs[i]/2
        fire_r = r[i]

        if fire_r < fire_width2:  # receiver within fire panel
            # Find radiator dimensions
            l_radiator_1 = fire_r + fire_width2
            l_radiator_2 = fire_r - fire_width2

            # Calculate phi for individual radiators
            p_radiator_1 = config_rectangle(l_radiator_1, room_height/2, receiver_distance)
            p_radiator_2 = config_rectangle(l_radiator_2, room_height/2, receiver_distance)

            # Sum all phi
            phi[i] = p_radiator_1 + p_radiator_2

        elif fire_r > fire_width2:  # receiver outside fire panel
            # Find radiator dimensions
            l_cavity = fire_r - fire_width2
            l_radiator = fire_r + fire_width2

            # Calculate phi for individual radiators
            p_radiator = config_rectangle(l_radiator, room_height/2, receiver_distance)
            p_cavity = config_rectangle(l_cavity, room_height/2, receiver_distance)

            # Sum all phi
            phi[i] = p_radiator - p_cavity

        elif fire_r == fire_width2:  # receiver on the edge of fire panel
            # Find radiator dimensions
            l_radiator = fire_width2 * 2

            # Calculate phi for individual radiators
            p_radiator = config_rectangle(l_radiator, room_height/2, receiver_distance)

            # Sum all phi
            phi[i] = p_radiator



    phi *= 2.

    import matplotlib.pyplot as plt
    plt.plot(c, phi)
    plt.grid(True)
    plt.show()
