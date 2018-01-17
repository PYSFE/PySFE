import numpy as np
import matplotlib.pyplot as plt

def config_rectangle(v1, v2, v3):
    # configuration factor for a parellel source and reciever
    # = config_rectangle(width [v1], height [v2], distance [v3])

    w = v1
    h = v2
    y = v3

    # intermediate values. Calculation from FRS Technical Paper 2

    if h > 0:  # Then
        s = w / h
        alpha = w * h / y / y
        c1 = np.sqrt(alpha * s / (1 + alpha * s))
        c2 = np.sqrt(alpha / s / (1 + alpha * s))
        c3 = np.sqrt(alpha / s / (1 + alpha / s))
        c4 = np.sqrt(alpha * s / (1 + alpha / s))
        c5 = c1 * np.arctan(c2) + c3 * np.arctan(c4)
        #   Configuration factor
        phi = c5 / (2*np.pi)
    else:
        phi = 0  #
    return phi

if __name__=='__main__':
    width = 1
    height = 5
    distance = 2
    phi = config_rectangle(width/2,height/2,distance)*4
    print(phi)

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

plt.plot(mid_out,phi_out)
plt.grid(True)
plt.show()