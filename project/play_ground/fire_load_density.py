# -*- coding: utf-8 -*-
import numpy as np

x = np.linspace(1,56,5600)

sigma_=833
miu_=747

y = 1 / (x * sigma_ * np.sqrt(2*np.pi)) * np.exp(-.5*((np.log(x)-miu_)/sigma_)**2)
# y = np.gradient(y,x)

import matplotlib.pyplot as plt
plt.plot(x,y)
plt.show()

