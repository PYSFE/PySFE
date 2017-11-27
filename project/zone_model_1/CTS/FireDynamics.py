import numpy as np
from sympy.solvers import solve
from sympy import Symbol
from scipy.interpolate import interp1d
from scipy.integrate import trapz


def steel_specific_heat_carbon_steel(temperature):
    """
        DESCRIPTION:
            [BS EN 1993-1-2:2005, 3.4.1.2]
            Calculate steel specific heat according to temperature
        PARAMETERS:
            temperature     {float, K}          Given temperature

            __return__      {float, J/kg/K}     Specific heat
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


def thermal_conductivity_carbon_steel(temperature):
    """
        DESCRIPTION:
            [BS EN 1993-1-2:2005, 3.4.1.3]
        PARAMETERS:
        OUTPUTS:
        REMARKS:
    """
    temperature += 273.15
    if 20 <= temperature <= 800:
        return 54 - 0.0333 * temperature
    elif 800 <= temperature <= 1200:
        return 27.3
    else:
        return 0


def reduction_factor_carbon_steel(steel_temperature):
    """
    DESCRIPTION:
        [BS EN 1993-1-2:2005, Table 3.1]
        Return reductions factors given temperature.
    PARAMETERS:
        temperature     {float, K}  Steel temperature

        self            {tuple, -}  (k_y_theta, k_p_theta, k_E_theta)
        k_y_theta       {float, /}  reduction factor for effective yield strength
        k_p_theta       {float, /}  reduction factor for proportional limit
        k_E_theta       {float, /}  reduction factor for the slope of the linear elastic range
    REMARKS:
        1.  293 [K] < steel_temperature < 1473 [K]
    """
    steel_temperature -= 273.15  # convert from [K] to [C], as the data below is in [C]
    temperature = [20,100,200,300,400,500,600,700,800,900,1000,1100,1200]
    k_y_theta = [1,1,1,1,1,0.78,0.47,0.23,0.11,0.06,0.04,0.02,0]
    k_p_theta = [1,1,0.807,0.613,0.42,0.36,0.18,0.075,0.05,0.0375,0.025,0.0125,0]
    k_E_theta = [1,1,0.9,0.8,0.7,0.6,0.31,0.13,0.09,0.0675,0.045,0.0225,0]

    k_y_theta = np.interp(steel_temperature,temperature,k_y_theta)
    k_p_theta = np.interp(steel_temperature,temperature,k_p_theta)
    k_E_theta = np.interp(steel_temperature,temperature,k_E_theta)

    return k_y_theta, k_p_theta, k_E_theta


def relative_thermal_elongation_carbon_steel(steel_temperature):
    steel_temperature -= 273.15
    if 20 <= steel_temperature < 750:
        relative_thermal_elongation = 1.2e-5 * steel_temperature + 0.4e-8 * np.power(steel_temperature,2) - 2.416e-4
    elif 750 <= steel_temperature <= 860:
        relative_thermal_elongation = 1.1e-2
    elif 860 < steel_temperature <= 1200:
        relative_thermal_elongation = 2e-5 * steel_temperature - 6.2e-3
    else:
        relative_thermal_elongation = None
    return relative_thermal_elongation


def stress_strain_elevated_temperature_carbon_steel(steel_strain, stress_proportional, stress_yield, elastic_modulus):
    """
        DESCRIPTION:
            [BS EN 1993-1-2:2005, Clause 3.2.2]
            Calculates steel stress for a given strain.
        PARAMETERS:
            VARIABLE                    NOTATION        {TYPE, UNIT}    DESCRIPTION
            steel_strain                eps             {double, /}     Current steel strain, for calculating its
                                                                        corresponding stress
            strain_proportional_limit   eps_p_theta     {double, /}     e_p_theta - Strain at the proportional limit
            strain_yield_strain         eps_y_theta     {double, /}     e_y_theta - Yield strain
            strain_yield_strength       eps_t_theta     {double, /}     e_t_theta - Limiting strain for yield strength
            strain_ultimate             eps_u_theta     {double, /}     e_u_theta - Ultimate strain
        REMARKS:
    """
    e = steel_strain

    f1 = stress_proportional
    f2 = stress_yield

    e1 = f1/elastic_modulus
    e2 = 0.02
    e3 = 0.15
    e4 = 0.20

    c = np.power(f2-f1, 2) / ((e2-e1) * elastic_modulus - 2 * (f2-f1))
    b = np.sqrt(c * (e2-e1) * elastic_modulus + np.power(c, 2))
    a = np.sqrt((e2-e1) * (e2 - e1 + c / elastic_modulus))

    if e <= e1:
        stress = e * elastic_modulus
        elastic_modulus = elastic_modulus
    elif e1 < e < e2:
        stress = f1 - c + (b / a) * np.sqrt(np.power(a,2) - np.power(e2 - e,2))
        elastic_modulus = (b * (e2 - e)) / (a * np.sqrt(np.power(a,2) - np.power(e2 - e, 2)))
    elif e2 <= e <= e3:
        stress = stress_yield
        elastic_modulus = 0
    elif e3 < e < e4:
        stress = f2 * (1 - (e - e3) / (e4 - e3))
        elastic_modulus = None
    elif e == e4:
        stress = 0
        elastic_modulus = None
    else:
        stress = None
        elastic_modulus = None

    return stress


def lookup_weighted_average(x1, x2, x, y):
        if x2 is None or x1 == x2:
            y = interp1d(x, y)(x1)

        else:
            # make sure x1 is always the smallest value
            if x1 > x2:
                x1 += x2
                x2 = x1 - x2
                x1 -= x2

            # get y1 and y2 based on x1 and x2
            y1 = interp1d(x, y)(x1)
            y2 = interp1d(x, y)(x2)

            # stripe the T and c array for only in the range of temp1 and temp2
            mask_arr = (x > x1) * (x < x2)
            x = x[mask_arr]
            y = y[mask_arr]

            # including the input boundary values, i.e. temp1 and temp2, c1 and c2
            x = np.concatenate(([x1], x, [x2]))
            y = np.concatenate(([y1], y, [y2]))

            # calculate average value via integration
            y = trapz(y, x) / (x2 - x1)

        return float(y)


def reduction_factor_buckling_fire_carbon_steel(stress_yield, lambda_, k_y_theta, k_E_theta):
    stress_yield /= 1e6
    lambda_theta = lambda_ * np.sqrt(k_y_theta / k_E_theta)
    a = 0.65 * np.sqrt(235/stress_yield)
    phi_theta = 0.5 * (1 + a * lambda_theta + np.power(lambda_theta,2))
    chi_fi = 1 / (phi_theta + np.sqrt(np.power(phi_theta,2) - np.power(lambda_theta,2)))
    if chi_fi > 1:
        print("something")
    return chi_fi


class DensityRatio(object):  # constant pressure (1atm)
    def __init__(self, material_str):
        # 'data_dict' format. {material_name: (temperature_array, thermal_conductivity_array)}
        data_dict = {
            "timber ec5-1-2": (
                # [K]
                [293.33,	372.23,	394.25,	474.98,	524.53,	574.07,	623.61,	673.15,	873.15,	1074.98, 1500.00,   5000],
                # [wet/dry density]
                [1.120,	    1.120,	1.000,	1.000,	0.923,	0.759,	0.521,	0.378,	0.277,	0.254,	 0.003,     0]
            )
        }
        for key in data_dict:
            data_dict[key] = [np.asarray(data_dict[key][0]), np.asarray(data_dict[key][1])]

        self.T, self.rho = data_dict[material_str]
        self.accumulative_average = None

    def temp(self, temperature1_kelvin, temperature2_kelvin=None):
        return lookup_weighted_average(temperature1_kelvin, temperature2_kelvin, self.T, self.rho)


class SpecificHeatP(object):  # constant pressure (1atm)
    def __init__(self, material_str):
        # 'data_dict' format. {material_name: (temperature_array, thermal_conductivity_array)}
        data_dict = {
            # [K]
            "N2": (np.asarray([175, 200, 225, 250, 275, 300, 325, 350, 375, 400,
                               450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
                               950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400,
                               1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400,
                               2500, 2600, 2700, 2800, 2900, 3000, 3500, 4000, 4500, 5000,
                               5500, 6000]),
                   # [J/kg/K]
                   np.asarray([1.039, 1.039, 1.039, 1.039, 1.039, 1.040, 1.040, 1.041, 1.042, 1.044,
                               1.049, 1.056, 1.065, 1.075, 1.086, 1.098, 1.110, 1.122, 1.134, 1.146,
                               1.157, 1.167, 1.177, 1.187, 1.196, 1.204, 1.212, 1.219, 1.226, 1.232,
                               1.244, 1.254, 1.263, 1.271, 1.278, 1.284, 1.290, 1.295, 1.300, 1.304,
                               1.307, 1.311, 1.314, 1.317, 1.320, 1.323, 1.333, 1.342, 1.349, 1.355,
                               1.362, 1.369,]) * 1000.),

            # [K]
            "O2": (np.asarray([175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 450,
                               500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000,
                               1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1500, 1600,
                               1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600,
                               2700, 2800, 2900, 3000, 3500, 4000, 4500, 5000, 5500, 6000,]),
                   # [J/kg/K]
                   np.asarray([0.910, 0.910, 0.911, 0.913, 0.915, 0.918, 0.923, 0.928, 0.934, 0.941,
                               0.956, 0.972, 0.988, 1.003, 1.017, 1.031, 1.043, 1.054, 1.065, 1.074,
                               1.082, 1.090, 1.097, 1.103, 1.109, 1.115, 1.120, 1.125, 1.130, 1.134,
                               1.143, 1.151, 1.158, 1.166, 1.173, 1.181, 1.188, 1.195, 1.202, 1.209,
                               1.216, 1.223, 1.230, 1.236, 1.243, 1.249, 1.276, 1.299, 1.316, 1.328,
                               1.337, 1.344,]) * 1000.),
            "CO2": (np.asarray([175, 200, 225, 250, 275, 300, 325, 350, 375, 400,
                                450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
                                950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400,
                                1500, 1600, 1700, 1800, 1900, 2000]),
                    # [J/kg/K]
                    np.asarray([0.709, 0.735, 0.763, 0.791, 0.819, 0.846, 0.871, 0.895, 0.918, 0.939,
                                0.978, 1.014, 1.046, 1.075, 1.102, 1.126, 1.148, 1.168, 1.187, 1.204,
                                1.220, 1.234, 1.247, 1.259, 1.270, 1.280, 1.290, 1.298, 1.306, 1.313,
                                1.326, 1.338, 1.348, 1.356, 1.364, 1.371]) * 1000.),

            # Stewart, R. B. et al (1988) - Thermodynamic properties of Argon from the triple point to 1200K
            # [K]
            "Ar": (np.asarray([110, 150, 200, 250, 320, 360, 470, 740, 6000]),
                   np.asarray([21.63, 21.02, 20.89, 20.85, 20.82, 20.81, 20.80, 20.79, 20.79]) / 39.948 * 1000),

            # http://www.engineeringtoolbox.com/water-vapor-d_979.html
            # [K]
            "H2O": (np.asarray([175, 200, 225, 250, 275, 300, 325, 350, 375, 400,
                                450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
                                950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400,
                                1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400,
                                2500, 2600, 2700, 2800, 2900, 3000, 3500, 4000, 4500, 5000,
                                5500, 6000]),
                    # [J/kg/K]
                    np.asarray([1.850, 1.851, 1.852, 1.855, 1.859, 1.864, 1.871, 1.880, 1.890, 1.901,
                                1.926, 1.954, 1.984, 2.015, 2.047, 2.080, 2.113, 2.147, 2.182, 2.217,
                                2.252, 2.288, 2.323, 2.358, 2.392, 2.425, 2.458, 2.490, 2.521, 2.552,
                                2.609, 2.662, 2.711, 2.756, 2.798, 2.836, 2.872, 2.904, 2.934, 2.962,
                                2.987, 3.011, 3.033, 3.053, 3.072, 3.090, 3.163, 3.217, 3.258, 3.292,
                                3.322, 3.350,
                                ]) * 1000.),

            "C3H8": (np.asarray([0, 5000]),
                     np.asarray([0, 5000])),

            "CH4": (np.asarray([0, 5000]),
                    np.asarray([0, 5000])),

            "timber ec5-1-2": (
                [293.15, 372.15, 372.15, 393.15, 393.15, 473.15, 523.15, 573.15, 623.15, 673.15, 873.15, 1073.15, 1473.15],
                [1530.,  1770.,  13600., 13500., 2120.,  2000.,  1620.,  710.,   850.,   1000.,  1400.,  1650.,   1650.]
            )
        }
        for key in data_dict:
            data_dict[key] = [np.asarray(data_dict[key][0]), np.asarray(data_dict[key][1])]

        if 'constant' in material_str:
            self.T = (0., 5000.)
            self.c = (float(material_str.split('_')[1]), float(material_str.split('_')[1]))
        else:
            self.T, self.c = data_dict[material_str]

        self.accumulative_average = None

    def temp(self, temperature1_kelvin, temperature2_kelvin=None):
        return lookup_weighted_average(temperature1_kelvin, temperature2_kelvin, self.T, self.c)


class ThermalConductivity(object):
    def __init__(self, material_str):
        # 'data_dict' format. {material_name: (temperature_array, thermal_conductivity_array)}
        data_dict = {
            # obtained from http://www.engineeringtoolbox.com/thermal-conductivity-d_429.html, range from 200C to 1200C
            "fire brick": ([0, 473.15, 673.15, 873.15, 1073.15, 1273.15, 1473.15, 5000],
                           [0.27, 0.27, 0.27, 0.29, 0.31, 0.33, 0.35, 0.35]),
            "fire brick jm32": ((0000.00, 0673.15, 0873.15, 1073.15, 1273.15, 1473.15, 1673.15),
                                (0000.49, 0000.49, 0000.50, 0000.51, 0000.53, 0000.56, 0000.60)),
            "fire brick jm30": ((0000.00, 0673.15, 0873.15, 1073.15, 1273.15, 1473.15, 1673.15),
                                (0000.38, 0000.38, 0000.39, 0000.40, 0000.41, 0000.42, 0000.42)),
            "fire brick jm28": ((0000.00, 0673.15, 0873.15, 1073.15, 1273.15, 1473.15, 1673.15),
                                (0000.30, 0000.30, 0000.32, 0000.34, 0000.36, 0000.38, 0000.38)),
            "glass": ([],
                      []),
            "timber ec5-1-2": ([293.15, 473.15, 623.15, 773.15, 1073.15, 1473.15],  # [K]
                               [0.12,   0.15,   0.07,   0.09,   0.35,    1.50])  # [W/m/K]
        }
        for key in data_dict:
            data_dict[key] = [np.asarray(data_dict[key][0]), np.asarray(data_dict[key][1])]

        if 'constant' in material_str:
            self.T = (0., 5000.)
            self.k = (float(material_str.split(' ')[1]), float(material_str.split(' ')[1]))
        else:
            self.T, self.k = data_dict[material_str]

    def temp(self, temperature1_kelvin, temperature2_kelvin=None):
        return lookup_weighted_average(temperature1_kelvin, temperature2_kelvin, self.T, self.k)


class gaseous_chamber_ambient_pressure(object):
    def __init__(self, gases_molar_mass_dict, array_length_int):

        # primariy properties
        self._array_length_int = array_length_int
        self._gases_molar_mass = gases_molar_mass_dict

        # property containers
        self._gases_mass = {gas: np.zeros((array_length_int,), float) for gas in self._gases_molar_mass}
        self._gases_mass_scaler = np.zeros((array_length_int,), float)
        self._gases_mole = {gas: np.zeros((array_length_int,), float) for gas in self._gases_molar_mass}
        self._gases_mole_scaler = np.zeros((array_length_int,), float)
        self._gases_cp_obj = {gas: SpecificHeatP(gas) for gas in self._gases_molar_mass}  # the c object is incomplete
        self._gases_cp = {gas: np.zeros((array_length_int,), float) for gas in self._gases_molar_mass}  # storage only
        self._gases_energy = {gas: np.zeros((array_length_int,), float) for gas in self._gases_molar_mass}
        self._gases_energy_scaler = np.zeros((array_length_int,), float)

        # set temperature
        self._temperature = np.zeros((array_length_int,), float)

    def set_mass(self, gas_str, mass_kg_float, i_int):
        # convert from [kg] to [mol]
        mole_mol_float = mass_kg_float / self._gases_molar_mass[gas_str]

        # calculate total mass and mole (replace the previous value with the new value)
        self._gases_mass_scaler[i_int] += (mass_kg_float - self._gases_mass[gas_str][i_int])
        self._gases_mole_scaler[i_int] += (mole_mol_float - self._gases_mole[gas_str][i_int])

        # assign values in [kg] and [mol]
        self._gases_mass[gas_str][i_int] = mass_kg_float
        self._gases_mole[gas_str][i_int] = mole_mol_float

    def set_mole(self, gas_str, mole_mol_float, i_int):
        # convert from [mol] to [kg]
        mass_kg_float = mole_mol_float * self._gases_molar_mass[gas_str]

        # calculate total mass and mole (replace the previous value with the new value)
        self._gases_mass_scaler[i_int] += mass_kg_float - self._gases_mass[gas_str][i_int]
        self._gases_mole_scaler[i_int] += mole_mol_float - self._gases_mole[gas_str][i_int]

        # assgin values in [kg] and [mol]
        self._gases_mass[gas_str][i_int] = mass_kg_float
        self._gases_mole[gas_str][i_int] = mole_mol_float

    def set_mass_by_proportion(self, gas_str, mass_kg, dict_proportion_data, i_int):
        total_mass = mass_kg / dict_proportion_data[gas_str]
        for key in dict_proportion_data:
            self.set_mass(key, total_mass * dict_proportion_data[key], i_int)

    def set_mole_by_proportion(self, gas_str, mole_mol, dict_proportion_data, i_int):
        total_mole = mole_mol / dict_proportion_data[gas_str]
        for key in dict_proportion_data:
            self.set_mole(key, total_mole * dict_proportion_data[key], i_int)

    def get_mass(self, gas_str=None, i_int=None):
        if gas_str is None:
            return self._gases_mass
        elif i_int is None:
            return self._gases_mass[gas_str]
        else:
            return self._gases_mass[gas_str][i_int]

    def get_mole(self, gas_str=None, i_int=None):
        if gas_str is None:
            return self._gases_mole
        elif i_int is None:
            return self._gases_mole[gas_str]
        else:
            return self._gases_mole[gas_str][i_int]

    def get_mass_total(self, i_int=None):
        if i_int is None:
            return self._gases_mass_scaler
        else:
            return self._gases_mass_scaler[i_int]

    def get_mole_total(self, i_int=None):
        if i_int is None:
            return self._gases_mole_scaler
        else:
            return self._gases_mole_scaler[i_int]

    def get_content_mass(self, gas_str=None, i_int=None):
        if gas_str is None:
            return {key: self._gases_mass[key] / self._gases_mass_scaler for key in self._gases_mass}
        elif i_int is None:
            return self._gases_mass[gas_str] / self._gases_mass_scaler
        else:
            return self._gases_mass[gas_str][i_int] / self._gases_mass_scaler[i_int]

    def get_content_mole(self, gas_str=None, i_int=None):
        if gas_str is None:
            return {key: self._gases_mole[key] / self._gases_mole_scaler for key in self._gases_mole}
        elif i_int is None:
            return self._gases_mole[gas_str] / self._gases_mole_scaler
        else:
            return self._gases_mole[gas_str][i_int] / self._gases_mole_scaler[i_int]

    def calc_energy_for_temperature_raise(self, T_0, T_1, i_int):
        dT = T_1 - T_0
        energy = 0.
        for key in self._gases_mass:
            self._gases_cp[key][i_int] = self._gases_cp_obj[key].temp(T_0, T_1)
            cp = self._gases_cp[key][i_int]
            m = self._gases_mass[key][i_int]
            energy += cp * m * dT

        return energy
