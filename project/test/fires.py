import matplotlib.pyplot as plt
from project.func.temperature_fires import travelling_fire

time, temperature, data = travelling_fire(
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
