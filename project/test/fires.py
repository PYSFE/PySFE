import matplotlib.pyplot as plt

def travelling_fire():
    from project.func.temperature_fires import travelling_fire as fire
    time, temperature, data = fire(
        T_0=293.15,
        q_fd=900e6,
        RHRf=0.15e6,
        l=150,
        w=17.4,
        s=0.012,
        h_s=3.5,
        l_s=105,
        time_step=60,
        time_ubound=22080
    )

def parametric_fire():
    from project.func.temperature_fires import parametric_eurocode1 as fire
    return fire(total_enclosure_area=2205.348,
                floor_area=853.187,
                opening_area=105.684078931808,
                opening_height=2,
                density_boundary=600,
                specific_heat_boundary=5425.347,
                thermal_conductivity_boundary=0.12,
                fire_load_density_floor=430.108,
                fire_growth_rate=20*60,
                time_step=0.1,
                time_extend=600)

def parametric_fire2():
    from project.lhs_max_st_temp.ec_param_fire import param_fire as fire
    return fire(dim1=,
                dim2=,
                dim3=,
                op1dim=,
                op2dim=,
                per_op=,
                qfd=,
                tlim=,
                b=,
                duration=,
                dt=)


plt.plot(time, temperature)
plt.show()
