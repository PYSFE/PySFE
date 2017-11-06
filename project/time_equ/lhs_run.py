from pyDOE import *
from scipy.stats.distributions import norm, gumbel_r, gumbel_l
import numpy as np
import project.time_equ.ec_param_fire as pf
import project.time_equ.ec3_ht as ht
import matplotlib.pyplot as plt
from project.func.temperature_fires import travelling_fire as fire
from project.func.temperature_steel_section import protected_steel_eurocode as steel
from project.dat.steel_carbon import Thermal as steel_thermal_dat

#   Handy interim functions

def linear_dist(min, max, prob):
    lin_invar = ((max-min)* prob)+min
    return lin_invar

def get_max_st_temp(tsec,temps,rho,c,Ap,kp,rhop,cp,dp,Hp):
    # data_dict_out, time, tempsteel, temperature_rate_steel, specific_heat_steel = ht.make_temperature_eurocode_protected_steel(
    #      tsec,temps,rho,c,Ap,kp,rhop,cp,dp,Hp)
    time, temperature, data_all = steel(
        time=tsec,
        temperature_ambient=temps,
        rho_steel_T=rho,
        c_steel_T=c,
        area_steel_section=Ap,
        k_protection=kp,
        density_protection=rhop,
        c_protection=cp,
        thickness_protection=dp,
        perimeter_protected=Hp
    )

    max_temp = np.amax(temperature)
    return max_temp, temperature


steel_prop = steel_thermal_dat()
c = steel_prop.property_thermal_specific_heat()
rho = steel_prop.property_density()

#   Define the inputs
lhs_iterations = 1000

#   Compartment dimensions
breadth = 20
depth = 40
height = 3.0
win_width = 20
win_height = 2.5

#   Deterministic fire inputs
t_start = 0
limit_time = 0.333
inertia =720
fire_dur = 20000
time_step = 30
hrr_pua = 0.25

#   Section properties

Hp = 0.1
Ap = 1.6e-03
dp = 0.0125
kp = 0.12
rhop = 600
cp = 1200

#   Set distribution mean and standard dev
qfd_std = 234
qfd_mean = 780
glaz_min = 0.1
glaz_max = 0.999
beam_min = 0.6
beam_max = 0.9999
com_eff_min = 0.7
com_eff_max = 1.0
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
std_nft = (1.939 - (np.log(avg_nft)*0.266))

#   Convert LHS probabilities to distribution invariants
comb_lhs = linear_dist(com_eff_min,com_eff_max,rnd_array_4)
qfd_lhs = gumbel_r(loc=qfd_loc,scale=qfd_scale).ppf(rnd_array_1)*comb_lhs
glaz_lhs = linear_dist(glaz_min, glaz_max, rnd_array_2)
beam_lhs = linear_dist(beam_min, beam_max, rnd_array_3) * depth
spread_lhs = linear_dist(spread_min, spread_max, rnd_array_5)

#   initialise output arrays
peak_st_fract = []
peak_st_temp = []

for i in range(0,lhs_iterations):
    fled = qfd_lhs[i]
    open_frac = glaz_lhs[i]
    spread = spread_lhs[i]
    beam_pos = beam_lhs[i]

    #   Get parametric fire curve
    #tsec, tmin, temps = pf.param_fire(breadth, depth, height, win_width, win_height, open_frac, fled, limit_time, inertia, fire_dur,
                                   #time_step)

    #   Get travelling fire curve
    print("iteration =", i+1, fled,
        hrr_pua,
        depth,
        breadth,
        spread,
        height,
        beam_pos,
        t_start,
        fire_dur,
        time_step)

    tsec, temps, data_ = fire(
        T_0=293.15,
        q_fd=fled * 1e6,
        RHRf=hrr_pua * 1e6,
        l=depth,
        w=breadth,
        s=spread,
        h_s=beam_pos,
        l_s=height,
        time_step=time_step,
        time_ubound=fire_dur,
    )

    max_temp, steelt = get_max_st_temp(tsec,temps,rho,c,Ap,kp,rhop,cp,dp,Hp)
    st_fract = 1- (i / lhs_iterations)
    peak_st_temp.append(max_temp)
    peak_st_fract.append(st_fract)

peak_st_temp = np.sort(peak_st_temp)

# Plot outputs
plt.figure(1)
plt.plot(peak_st_temp,peak_st_fract)
plt.grid(True)
plt.xlabel('Max Steel Temperature [Deg C]')
plt.ylabel('Probability of not exceeding temperature [-]')
plt.show()

plt.figure(2)
plt.plot(tsec,temps)
plt.plot(tsec,steelt)
plt.show()

print("Stop")

