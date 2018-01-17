import numpy as np
import matplotlib.pyplot as plt
import warnings


def config_rectangle(width, height, distance):
    # configuration factor for a parellel source and reciever
    # = config_rectangle(width [width], room_height [room_height], distance [distance])

    w, h, y = width, height, distance

    # intermediate values. Calculation from FRS Technical Paper 2

    if h == 0. or w == 0. or y == 0.:
        warnings.warn("In valid input variables: width={}, room_height={}, distance={}. "
                      "Zero view factor is returned.".format(h, w, y))
        phi = 0.
    else:
        s = w / h
        alpha = w * h / y / y
        c1 = np.sqrt(alpha * s / (1 + alpha * s))
        c2 = np.sqrt(alpha / s / (1 + alpha * s))
        c3 = np.sqrt(alpha / s / (1 + alpha / s))
        c4 = np.sqrt(alpha * s / (1 + alpha / s))
        c5 = c1 * np.arctan(c2) + c3 * np.arctan(c4)
        #   Configuration factor
        phi = c5 / (2*np.pi)

    return phi


def view_factor_customised(fire_location, fire_width, receiver_location=1., receiver_distance=1., room_height=2.):

    # c = fire_location
    # r = fire_location - receiver_location

    phi = np.zeros_like(fire_location, dtype=float)

    for i, v in enumerate(fire_location):
        fire_width2 = fire_width[i]/2
        fire_r = np.abs(fire_location[i] - receiver_location)

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
        else:
            warnings.warn('fire_r = {}, fire_width2 = {}!'.format(fire_r, fire_width2))
            phi[i] = -1

    return phi * 2


if __name__ == '__main__':
    width = 1
    height = 5
    distance = 2
    print(config_rectangle(1, 5, 1) * 4)

    mid_out =[]
    phi_out=[]


    for mid in np.arange(0.1,10,0.01):
        location = 5
        r = np.abs(location - mid)
        if r > width/2:
            x1 = r + (width/2)
            x2 = r - (width/2)
            x2 = max(x2,0)
            phi1 = 4*config_rectangle(x1/2,height/2,distance)
            phi2 = 4*config_rectangle(x2/2,height/2,distance)
            phi3 = phi1-phi2
        elif r == (width /2):
            phi3 = config_rectangle(width,height/2,distance)*2
        else:
            x1 = r + (width / 2)
            x2 = (width / 2) - r
            x3 = x1 + x2
            phi1 = config_rectangle(x1,height/2,distance)
            phi2 = config_rectangle(x2, height/2,distance)
            phi3 = (2 * phi1) + (2*phi2)
        mid_out.append(mid)
        phi_out.append(phi3)

    print(phi_out)

    phi2 = view_factor_customised(fire_location=np.linspace(0, 10, 100),
                                  fire_width=np.ones(shape=(100,), dtype=float),
                                  receiver_location=4.5,
                                  receiver_distance=distance,
                                  room_height=height)

    plt.plot(mid_out,phi_out)
    plt.plot(np.linspace(0, 10, 100), phi2)
    plt.grid(True)
    # plt.show()