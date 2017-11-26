# -*- coding: utf-8 -*-


def travelling_fire():
    from project.func.temperature_fires import travelling_fire as fire
    import matplotlib.pyplot as plt
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
    plt.plot(time, temperature)
    plt.show()


def parametric_fire():
    from project.func.temperature_fires import parametric_eurocode1 as fire
    import copy

    kwargs = {"A_t": 360,
              "A_f": 100,
              "A_v": 72,
              "h_eq": 1,
              "q_fd": 600e6,
              "lambda_": 1,
              "rho": 1,
              "c": 2250000,
              "t_lim": 20*60,
              "time_end": 2*60*60,
              "time_step": 1,
              "time_start": 0,
              "temperature_initial": 293.15}

    import matplotlib.pyplot as plt
    for i in [72,50.4,36.000000001,32.4,21.6,14.4,7.2]:
        if i == 36:
            print("stop")
        kwargs["A_v"] = i
        x, y = fire(**copy.copy(kwargs))
        plt.plot(x/60, y-273.15, label="O={:03.2f}".format(kwargs["A_v"]*kwargs["h_eq"]**0.5/kwargs["A_t"]))
    plt.legend(loc=1)
    plt.show()
    return 0


if __name__ == "__main__":
    travelling_fire()
    parametric_fire()
