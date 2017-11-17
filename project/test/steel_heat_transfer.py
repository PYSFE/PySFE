def protected_steel():
    from project.func.temperature_steel_section import protected_steel_eurocode as steel
    from project.func.temperature_fires import standard_fire_iso834 as fire
    from project.dat.steel_carbon import Thermal
    import matplotlib.pyplot as plt
    import numpy as np

    steel_prop = Thermal()
    c = steel_prop.c()
    rho = steel_prop.rho()

    t, T = fire(np.arange(0,10080,60), 20+273.15)

    for d_p in np.arange(0.001, 0.2+0.02, 0.02):
        t_s, T_s, d = steel(
            time=t,
            temperature_ambient=T,
            rho_steel_T=rho,
            c_steel_T=c,
            area_steel_section=0.017,
            k_protection=0.2,
            rho_protection=800,
            c_protection=1700,
            thickness_protection=d_p,
            perimeter_protected=2.14
        )
        plt.plot(t_s, T_s, label="d_p={:5.3f}".format(d_p))

    plt.legend(loc=1)
    plt.show()

if __name__ == "__main__":
    protected_steel()