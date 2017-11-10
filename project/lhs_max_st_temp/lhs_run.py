from pyDOE import *
from scipy.stats.distributions import norm, gumbel_r, gumbel_l
import numpy as np
import project.lhs_max_st_temp.ec_param_fire as pf
import project.lhs_max_st_temp.ec3_ht as ht
import project.lhs_max_st_temp.tfm_alt as tfma
from project.dat.steel_carbon import Thermal
import matplotlib.pyplot as plt
from project.lhs_max_st_temp.lhs_run_mp import mc_calculation

steel_prop = Thermal()
c = steel_prop.c()
rho = steel_prop.rho()

#   Handy interim functions


def linear_dist(min, max, prob):
    lin_invar = ((max-min)* prob)+min
    return lin_invar

def get_max_st_temp(tsec,temps,rho,c,Ap,kp,rhop,cp,dp,Hp):
    data_dict_out, time, tempsteel, temperature_rate_steel, specific_heat_steel = ht.make_temperature_eurocode_protected_steel(
         tsec,temps,rho,c,Ap,kp,rhop,cp,dp,Hp)

    max_temp = np.amax(tempsteel)
    return max_temp, tempsteel


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
qfd_scale = (qfd_std*(6**0.5))/np.pi
qfd_loc = qfd_mean - (0.5722*qfd_scale)

#   Near field standard deviation
std_nft = (1.939 - (np.log(avg_nft)*0.266)) *avg_nft

#   Convert LHS probabilities to distribution invariants
comb_lhs = linear_dist(com_eff_min,com_eff_max,rnd_array_4)
qfd_lhs = gumbel_r(loc=qfd_loc,scale=qfd_scale).ppf(rnd_array_1)*comb_lhs
glaz_lhs = linear_dist(glaz_min, glaz_max, rnd_array_2)
beam_lhs = linear_dist(beam_min, beam_max, rnd_array_3) * depth
spread_lhs = linear_dist(spread_min, spread_max, rnd_array_5)
nft_lhs = norm(loc=avg_nft, scale=std_nft).ppf(rnd_array_6)

#   initialise output arrays
peak_st_fract = []
peak_st_temp = []

for i in range(0,lhs_iterations):
    dict_inputs = {
        "window_height":                win_height,
        "window_width":                 win_width,
        "window_open_fraction":         glaz_lhs[i],
        "room_breadth":                 breadth,
        "room_depth":                   depth,
        "room_height":                  height,
        "fire_load_density":            qfd_lhs[i],
        "fire_hrr_density":             hrr_pua,
        "fire_spread_speed":            spread_lhs[i],
        "time_limiting":                limit_time,
        "room_wall_thermal_inertia":    inertia,
        "fire_duration":                fire_dur,
        "time_step":                    time_step,
        "beam_position":                beam_lhs[i],
        "time_start":                   t_start,
        "temperature_max_near_field":   min([nft_lhs[i],1200]),
        "beam_rho":                     rho,
        "beam_c":                       c,
        "beam_cross_section_area":      Ap,
        "protection_k":                 kp,
        "protection_rho":               rhop,
        "protection_c":                 cp,
        "protection_depth":             dp,
        "protection_protected_perimeter": Hp
    }

    a = mc_calculation(**dict_inputs)


    fled = qfd_lhs[i]
    open_frac = glaz_lhs[i]
    spread = spread_lhs[i]
    beam_pos = beam_lhs[i]
    nft_c = min([nft_lhs[i],1200])

    #   Check on applicable fire curve

    av = win_height * win_width * open_frac
    af = breadth * depth
    at = (2 * af) + ((breadth + depth) * 2 * height)

    #   Opening factor - is it within EC limits?
    of_check = pf.opening_factor(av,win_height,at)

    #   Spread speed - Does the fire spread to involve the full compartment?
    sp_time = max([depth,breadth]) / spread
    burnout_m2 = max ([fled / hrr_pua, 900.])

    if sp_time < burnout_m2 and 0.02 < of_check <= 0.2: #   If fire spreads throughout compartment and ventilation is within EC limits = Parametric fire

        #   Get parametric fire curve
        tsec, tmin, temps = pf.param_fire(breadth, depth, height, win_width, win_height, open_frac, fled, limit_time, inertia, fire_dur,
                                   time_step)
        fmstr = "Parametric"

    else:   #   Otherwise, it is a travelling fire

        #   Get travelling fire curve
        tsec, temps, heat_release, distance_to_element = tfma.travelling_fire(
            fled,
            hrr_pua,
            depth,
            breadth,
            spread,
            height,
            beam_pos,
            t_start,
            fire_dur,
            time_step,
            nft_c,
            win_width,
            win_height,
            open_frac
        )
        fmstr = "Travelling"

    print("LHS_model_realisation_count =", i + 1, "Of", lhs_iterations)

    #   Optional unprotected steel code
    #tempsteel, temprate, hf, c_s = ht. make_temperature_eurocode_unprotected_steel(tsec,temps+273.15,Hp,Ap,0.1,7850,c,35,0.625)
    #tempsteel -= 273.15
    #max_temp = np.amax(tempsteel)

    #   Solve heat transfer using EC3 correlations
    max_temp, steelt = get_max_st_temp(tsec,temps,rho,c,Ap,kp,rhop,cp,dp,Hp)

    #   Create output arrays
    st_fract = ((i+1) / lhs_iterations)
    peak_st_temp.append(max_temp)
    peak_st_fract.append(st_fract)

#   Sort output array at the end to get CPD of steel temperatures

peak_st_temp = np.sort(peak_st_temp)

#   Write to csv

n_rows, = np.shape(peak_st_temp)
out = np.append(peak_st_temp,peak_st_fract,0)
out = np.reshape(out, (n_rows, 2),order='F')
np.savetxt("out.csv", out, delimiter=",")

# Plot outputs
plt.figure(1)
plt.plot(peak_st_temp,peak_st_fract)
plt.grid(True)
plt.xlabel('Max Steel Temperature [Deg C]')
plt.ylabel('Fractile [-]')
plt.show()

# plt.figure(2)
# plt.plot(tsec,temps)
# plt.plot(tsec,steelt)
# plt.show()

# plt.figure(3)
# plt.plot(tsec, heat_release)
# plt.show()

print("Stop")

