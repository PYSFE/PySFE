#   Parametric fire script

import numpy as np

#   Handy sub-calculations

#   Opening factor
def opening_factor(Av, heq, At):
    O = Av*np.sqrt(heq)/At
    return O

#   qtd
def qtd_calc(qfd,Af,At):
    qtd = qfd*Af/At
    return qtd

#   Max fire duration for ventilation control
def tmax_dur(qtd, O):
    tmax = 0.0002*qtd/O
    return tmax

#   Gamma (time modifier)
def gamma_calc(O,b):
    gamma = ((O/0.04)/(b/1160))**2
    return gamma

#   Fuel controlled opening factor
def Olim_calc(qtd,tlim):
    Olim = 0.0001 * qtd / tlim
    return Olim

#   Temperature during growth phase
def growth_temp(tstar):
    tgas = 20 + 1325*(1-0.324*np.exp(-0.2*tstar)-0.204*np.exp(-1.7*tstar)-0.472*np.exp(-19*tstar))
    return tgas

#   Decay branch 1
def decay_temp1(tstar, temp_max, tstarmax, x):
    tgas = temp_max - 625*(tstar-tstarmax*x)
    return tgas

#   Decay branch 2
def decay_temp2(tstar, temp_max, tstarmax, x):
    tgas = temp_max -250*(3-tstarmax)*(tstar-tstarmax*x)
    return tgas

#   Decay branch 1
def decay_temp3(tstar, temp_max, tstarmax, x):
    tgas = temp_max - 250*(tstar-tstarmax*x)
    return tgas

#   k modifier for specific combination of opening factor, qtd and inertia
def k_factor (O, qtd, b):
    k = 1 + (((O-0.04)/0.04)*((qtd-75)/75)*((1160-b)/1160))
    return k

#   Main parametric routine
def param_fire(dim1, dim2, dim3, op1dim, op2dim, glazf, qfd, tlim, b, duration, dt):
    #   Area Calculations
    Av = op1dim * op2dim * glazf
    heq = op2dim
    Af = dim1 * dim2
    At = (2*Af)+((dim1+dim2)*2*dim3)

    #   Preliminary factors
    open_fc = opening_factor(Av, heq, At)

    if open_fc < 0.02:
        open_fc = 0.02
    elif open_fc > 0.2:
        open_fc = 0.2

    qtd = qtd_calc(qfd, Af, At)
    tmax = tmax_dur(qtd, open_fc)
    gamma_orig = gamma_calc(open_fc, b)
    Olim = Olim_calc(qtd, tlim) #   limiting opening factor for fuel controlled case
    gammalim = gamma_calc(Olim, b) #    gamma lim for fuel controlled case
    tstar_max_c = min(gamma_orig*tmax, tlim) #  t* for cooling phase calculations

    tpmax = ((0.0002 * qtd) / open_fc) * gamma_orig


    #   Check on k factor for specific combination of factors
    if open_fc > 0.04 and qtd < 75 and b < 1160:
        k = k_factor(open_fc,qtd,b)
    else:
        k = 1

    # Confirm if fire is ventilation or fuel controlled
    if tmax < tlim:
        tmax = tlim
        gamma = gammalim * k
    else:
        gamma = gamma_orig

    # Calculate parametric time at peak temperature

    tstar_max = gamma * tmax

    # Calculate x

    if tmax > tlim:
        x = 1
    else:
        x = (tlim * gamma_orig) / tpmax

    #   Calculate maximum compartment temperature
    temp_max =growth_temp(tstar_max)

    #   Create time arrays
    t_sec = np.arange(0,duration+dt,dt)
    t_min = t_sec / 60
    t_hr = t_sec / 3600
    t_star = t_hr * gamma
    t_star_c = t_hr * gamma_orig

    # Cooling phase
    temp_grow = growth_temp(t_star)* (t_star < tstar_max).astype(int)

    temp_decay1 = decay_temp1(t_star_c,temp_max,tpmax, x) * (tpmax <= 0.5) * (t_star >= tstar_max)
    temp_decay1[temp_decay1<0] = 0
    temp_decay2 = decay_temp2(t_star_c,temp_max,tpmax, x) * (2 > tpmax > 0.5) * (t_star >= tstar_max)
    temp_decay2[temp_decay2<0] = 0
    temp_decay3 = decay_temp3(t_star_c,temp_max,tpmax, x) * (tpmax >= 2.0) * (t_star >= tstar_max)
    temp_decay3[temp_decay3<0] = 0
    temp_all = temp_grow+temp_decay1+temp_decay2+temp_decay3
    temp_all[temp_all<20] = 20

    return t_sec, t_min, temp_all