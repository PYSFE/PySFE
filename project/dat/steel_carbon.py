# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os


def thermal(property_name):
    """FUNCTION DESCRIPTION:
    This function is intended to simplify the process of obtaining
    :param property_name:
    :return:
    """
    dir_package = os.path.dirname(os.path.abspath(__file__))

    dir_files = {
        "density":
            "rho_1_T_steelc_ec.csv",  # BS EN 1993-1-2:2005, 3.2.2
        "thermal conductivity":
            "k_1_T_steelc_ec.csv",  # BS EN 1993-1-2:2005, 3.4.1.3
        "specific heat capacity":
            "c_1_T_steelc_ec.csv",  # BS EN 1993-1-2:2005, 3.4.1.2
        "reduction factor for the slope of the linear elastic range":
            "kE_1_T_steelc_ec.csv",  # BS EN 1993-1-2:2005, Table 3.1
        "reduction factor for proportional limit":
            "kp_1_T_steelc_ec.csv",  # BS EN 1993-1-2:2005, Table 3.1
        "reduction factor for effective yield strength":
            "ky_1_T_steelc_ec.csv"  # BS EN 1993-1-2:2005, Table 3.1
    }

    # read file
    data = pd.read_csv("/".join([dir_package, dir_files[property_name]]), delimiter=",", dtype=float)
    x, y = data.values[:, 0], data.values[:, 1]

    return interp1d(x, y)


class Thermal(object):
    @staticmethod
    def __make_property(property_name, dict_files=None):
        if dict_files is None: dict_files = {
            "rho": "rho_1_T_steelc_ec.csv",  # BS EN 1993-1-2:2005, 3.2.2
            "k": "k_1_T_steelc_ec.csv",  # BS EN 1993-1-2:2005, 3.4.1.3
            "c": "c_1_T_steelc_ec.csv",  # BS EN 1993-1-2:2005, 3.4.1.2
            "kE": "kE_1_T_steelc_ec.csv",  # BS EN 1993-1-2:2005, Table 3.1, (reduction factor, linear elastic)
            "kp": "kp_1_T_steelc_ec.csv",  # BS EN 1993-1-2:2005, Table 3.1, (reduction factor, proportional limit)
            "ky": "ky_1_T_steelc_ec.csv"  # BS EN 1993-1-2:2005, Table 3.1, (reduction factor, eff. yield str.)
        }
        file_name_enquired_property = dict_files[property_name]
        dir_this_folder = os.path.dirname(os.path.abspath(__file__))
        dir_file = "/".join([dir_this_folder, file_name_enquired_property])
        data_raw = pd.read_csv(dir_file, delimiter=",", dtype=float)
        # todo: fix the following line, rename database headers
        c1, c2 = data_raw.values[:, 0], data_raw.values[:, 1]
        return interp1d(c1, c2)

    def rho(self): return self.__make_property("rho")

    def k(self): return self.__make_property("k")

    def c(self): return self.__make_property("c")

    def kE(self): return self.__make_property("kE")

    def kp(self): return self.__make_property("kp")

    def ky(self): return self.__make_property("ky")

    def property_density(self): return self.rho()

    def property_thermal_conductivity(self): return self.k()

    def property_thermal_specific_heat(self): return self.c()

    def property_reduction_factor_kE(self): return self.kE()

    def property_reduction_factor_kp(self): return self.kp()

    def property_reduction_factor_ky(self): return self.ky()


def steel_specific_heat_carbon_steel(temperature):
    """
    :param temperature: {float} [K]
    :return: {float} [J/kg/K]
    """
    temperature -= 273.15
    if 20 <= temperature < 600:
        return 425 + 0.773 * temperature - 1.69e-3 * np.power(temperature, 2) + 2.22e-6 * np.power(temperature, 3)
    elif 600 <= temperature < 735:
        return 666 + 13002 / (738 - temperature)
    elif 735 <= temperature < 900:
        return 545 + 17820 / (temperature - 731)
    elif 900 <= temperature <= 1200:
        return 650
    else:
        return 0


if __name__ == "__main__":

    temperature = np.arange(20.,1200.+0.5,0.5) + 273.15
    prop = temperature * 0
    for i,v in enumerate(temperature):
        prop[i] = steel_specific_heat_carbon_steel(v)
    df = pd.DataFrame(
        {
            "Temperature [K]": temperature,
            "Specific Heat Capacity [J/kg/K]": prop
        }
    )
