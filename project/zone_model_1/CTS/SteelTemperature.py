# -*- coding: utf-8 -*-
import numpy as np
from CTS.FireDynamics import steel_specific_heat_carbon_steel
from CTS.FireDynamics import stress_strain_elevated_temperature_carbon_steel
import CTS.FireDynamics as fd_eurocode

class UnprotectedSteel(object):
    """
    DESCRIPTION:
    ATTRIBUTES:
            timeArray                           {ndarray, s}
            temperatureArray                    {ndarray, K}
            sectionPerimeter                    {m}
            sectionArea                         {m2}
            densitySteel                        {kg/m3}
            convectiveHeatTransferCoefficient   {W/m2/K}
            resultantEmissivity                 {-}
    REMARKS:
            1. Steel temperature is limited within 20C <= T_s <= 1200, due to specific heat
    """
    def __init__(self, timeArray, temperatureArray,
                 sectionPerimeter, sectionArea, boxPerimeter, boxArea,
                 densitySteel,
                 convectiveHeatTransferCoefficient, resultantEmissivity):
        # Assign attributes
        self.timeArray = timeArray
        self.temperatureArray = temperatureArray
        self.sectionPerimeter = float(sectionPerimeter)
        self.sectionArea = float(sectionArea)
        self.boxPerimeter = float(boxPerimeter)
        self.boxArea = float(boxArea)
        self.densitySteel = float(densitySteel)
        self.convectiveHeatTransferCoefficient = float(convectiveHeatTransferCoefficient)
        self.resultantEmissivity = float(resultantEmissivity)

        self.steelTemperatureArray = None
        self.steelTemperatureChangeArray = None
        self.heatFluxNet = None
        self.steelSpecificHeat = None

    def proceed(self):
        # Create steel temperature change array
        steelTemperatureChangeArray = np.zeros(np.shape(self.timeArray), dtype=float)
        steelTemperatureArray = np.zeros(np.shape(self.timeArray), dtype=float)
        heat_flux_net = np.zeros(np.shape(self.timeArray), dtype=float)
        steel_specific_heat = np.zeros(np.shape(self.timeArray), dtype=float)

        k_sh = 0.9 * (self.boxPerimeter / self.sectionArea) / (self.sectionPerimeter / self.sectionArea) # BS EN 1993-1-2:2005 (e4.26a)
        F = self.sectionPerimeter
        V = self.sectionArea
        rho_s = self.densitySteel
        h_c = self.convectiveHeatTransferCoefficient
        sigma = 56.7e-9
        epsilon = self.resultantEmissivity

        index = np.arange(1, np.size(self.timeArray), 1)
        steelTemperatureArray[0] = self.temperatureArray[0]
        for i in index:
            T_f = self.temperatureArray[i]
            T_s = steelTemperatureArray[i-1]
            steel_specific_heat[i] = float(steel_specific_heat_carbon_steel(steelTemperatureArray[i-1]))

            # BS EN 1993-1-2:2005 (e4.25)
            a = h_c * (T_f - T_s)
            b = sigma * epsilon * (np.power(T_f,4) - np.power(T_s,4))
            c = k_sh * F / V / rho_s / steel_specific_heat[i]
            d = self.timeArray[i] - self.timeArray[i-1]

            heat_flux_net[i] = a + b

            steelTemperatureChangeArray[i] = c * (a + b) * d
            steelTemperatureArray[i] = steelTemperatureArray[i-1] + steelTemperatureChangeArray[i]

        self.steelTemperatureArray = steelTemperatureArray
        self.steelTemperatureChangeArray = steelTemperatureChangeArray
        self.heatFluxNet = heat_flux_net
        self.steelSpecificHeat = steel_specific_heat

class ProtectedSteel_Buchanan(object):
    """
    DESCRIPTION:

    ATTRIBUTES:
        timeArray                       {ndarray, s}
        temperatureArray                {ndarray, K}
        sectionPerimeter                {m}
        sectionArea                     {m2}
        insulationThermalConductivity   {W/m/K}
        insulationDensity               {kg/m3}
        insulationSpecificHeat          {J/kg/K}
        insulationDepth                 {m}
    """
    def __init__(self, timeArray, temperatureArray,
                 sectionPerimeter, sectionArea,
                 insulationThermalConductivity, insulationDensity, insulationSpecificHeat, insulationDepth,
                 steelDensity):
        self.timeArray = timeArray
        self.temperatureArray = temperatureArray
        self.sectionPerimeter = float(sectionPerimeter)
        self.sectionArea = float(sectionArea)
        self.insulationThermalConductivity = float(insulationThermalConductivity)
        self.insulationDensity = float(insulationDensity)
        self.insulationSpecificHeat = float(insulationSpecificHeat)
        self.insulationDepth = float(insulationDepth)
        self.steelDensity = float(steelDensity)

        self.steelTemperatureArray = None
        self.steelTemperatureChangeArray = None

    def proceed(self):
        # Create steel temperature change array
        steelTemperatureChangeArray = np.zeros(np.shape(self.timeArray), dtype=float)
        steelTemperatureArray = np.zeros(np.shape(self.timeArray), dtype=float)

        # Assign variables for readability
        F = self.sectionPerimeter
        V = self.sectionArea
        d_i = self.insulationDepth
        k_i = self.insulationThermalConductivity
        rho_i = self.insulationDensity
        c_i = self.insulationSpecificHeat
        rho_s = self.steelDensity

        index = np.arange(1, np.size(self.timeArray), 1)
        steelTemperatureArray[0] = self.temperatureArray[0]
        for i in index:
            c_s = steel_specific_heat_carbon_steel(steelTemperatureArray[i-1])
            a = (F / V) * (k_i / d_i / rho_s / c_s)
            b = (rho_s * c_s) / (rho_s * c_s + (F / V) * (d_i * rho_i * c_i / 2))
            c = self.temperatureArray[i] - steelTemperatureArray[i-1]
            d = self.timeArray[i] - self.timeArray[i-1]
            steelTemperatureChangeArray[i] = a * b * c * d
            steelTemperatureArray[i] = steelTemperatureArray[i-1] + steelTemperatureChangeArray[i]

        self.steelTemperatureArray = steelTemperatureArray
        self.steelTemperatureChangeArray = steelTemperatureChangeArray

    @staticmethod
    def thermal_conductivity(temperature):
        T = temperature - 273.15
        if 20 <= T <= 800:
            return 54 - 0.0333 * T
        elif 800 <= T <= 1200:
            return 27.3

class ProtectedSteel_Eurocode(object):
    """
    DESCRIPTION:
        [BS EN 1993-1-2:2005]
        Calculates protected steel internal temperature, strength variation based on given time-temperature curve
    PARAMETERS:
        steel_yield_stress              f_y_theta   {float, N/m2}   Effective yield strength
        protectionArea                  A_p         {float, m2/m}   The appropriate area of fire protection material per
                                                                    unit length of the member.
        sectionVolume                   V           {float, m3/m}   The volume of the member per unit length.
        specificHeatSteel               c_a         {float, J/kg/K} Temperature dependent specific heat of steel.
        specificHeatProtection          c_p         {float, J/kg/K} Temperature independent specific heat of protection
                                                                    material.
        protectionDepth                 d_p         {float, m}      The thickness of the fire protection material.
        timeStep                        s           {float, s}      Time interval
        steelTemperature                T_a         {ndarray, C}    Steel temperature array, {float}.
        gasTemperature                  T_g         {ndarray, C}    Ambient temperature array, {float}.
        thermalConductivityProtection   lamda_p     {float, C}      Thermal conductivity of the fire protection system.
        densitySteel                    rho_a       {float, kg/m3}  Steel density.
        densityProtection               rho_p       {float, kg/m3}  Protection material density.
        gamma_m_fi                      -           {float, -}      Partial safety factor for fire situation
    REMARKS:

    """
    
    def __init__(self, time_array, temperature_array,
                 steel_yield_stress, steel_proportional_stress, steel_elastic_modulus,
                 section_volume, slenderness, density_steel,
                 specific_heat_protection, thermal_conductivity_protection,
                 density_protection, protection_depth, protection_area):
        self.timeArray = time_array
        self.temperatureArray = temperature_array
        self.steelYieldStress = float(steel_yield_stress)
        self.steelProportionalStress = float(steel_proportional_stress)
        self.steelElasticModulus = float(steel_elastic_modulus)
        self.sectionVolume = float(section_volume)
        self.slenderness = float(slenderness)
        self.densitySteel = float(density_steel)
        # self.memberLength = float(member_length)
        self.specificHeatProtection = float(specific_heat_protection)
        self.thermalConductivityProtection = float(thermal_conductivity_protection)
        self.densityProtection = float(density_protection)
        self.protectionDepth = float(protection_depth)
        self.protectionArea = float(protection_area)
        
        self.steelTemperatureArray = None
        self.steelTemperatureChangeArray = None
        self.steelSpecificHeat = None

        self.steelEffectiveYieldReductionFactor = None
        self.steelProportionalLimitReductionFactor = None
        self.steelElasticModulusReductionFactor = None
        self.steelSlendernessTemperatureDependent = None

        self.steelStrength = None
        self.steelThermalStrain = None
        self.steelThermalStress = None
    
    def proceed(self):
        self.get_steel_temperature()
        self.get_steel_properties_variation()

    def get_steel_temperature(self):
        # --------------------------------------------------------------------------------------------------------------
        # steel temperature calculation procedure
        # --------------------------------------------------------------------------------------------------------------
        V = self.sectionVolume
        rho_a = self.densitySteel
        lamda_p = self.thermalConductivityProtection
        rho_p = self.densityProtection
        d_p = self.protectionDepth
        A_p = self.protectionArea
        c_p = self.specificHeatProtection

        steelTemperatureArray = np.zeros(np.shape(self.timeArray),dtype=float)
        steelTemperatureChangeArray = np.zeros(np.shape(self.timeArray),dtype=float)
        steelSpecificHeat = np.zeros(np.shape(self.timeArray),dtype=float)

        steelTemperatureArray[0] = self.temperatureArray[0]  # initial steel temperature is equel to ambient
        iterTemperatureArray = iter(self.temperatureArray)  # skip the first item
        next(iterTemperatureArray)  # skip the first item
        for i, currentGasTemperature in enumerate(iterTemperatureArray):
            i += 1 # the first item had been skipped.

            steelSpecificHeat[i] = steel_specific_heat_carbon_steel(temperature=steelTemperatureArray[i-1]+273.15)

            # Steel temperature equation followed [BS EN 1993-1-2:2005, Clauses 4.2.5.2]

            phi = (c_p * rho_p / steelSpecificHeat[i] / rho_a) * d_p * A_p / V

            a = (lamda_p * A_p / V) / (d_p * steelSpecificHeat[i] * rho_a)
            b = (currentGasTemperature - steelTemperatureArray[i-1]) / (1 + phi/3)
            c = (np.exp(phi/10)-1) * (currentGasTemperature - self.temperatureArray[i-1])
            d = self.timeArray[i] - self.timeArray[i-1]

            steelTemperatureChangeArray[i] = a * b * d - c

            if steelTemperatureChangeArray[i] < 0 and currentGasTemperature - steelTemperatureArray[i-1] > 0:
                steelTemperatureChangeArray[i] = 0
            steelTemperatureArray[i] = steelTemperatureArray[i-1] + steelTemperatureChangeArray[i]

        self.steelTemperatureArray = steelTemperatureArray
        self.steelTemperatureChangeArray = steelTemperatureChangeArray
        self.steelSpecificHeat = steelSpecificHeat

    def get_steel_properties_variation(self):
        # --------------------------------------------------------------------------------------------------------------
        # steel strength variation according to temperature
        # --------------------------------------------------------------------------------------------------------------

        # variable assignment - for strength calculation
        section_area = self.sectionVolume
        stress_yield = self.steelYieldStress
        k_y_theta = np.zeros(shape = np.shape(self.timeArray), dtype=float)
        k_p_theta = np.zeros(shape = np.shape(self.timeArray), dtype=float)
        k_E_theta = np.zeros(shape = np.shape(self.timeArray), dtype=float)
        lamda_theta = np.zeros(shape = np.shape(self.timeArray), dtype=float)

        # variable assignment - for thermal stress calculation
        strain_theta = np.zeros(shape = np.shape(self.timeArray), dtype=float)
        stress_theta = np.zeros(shape = np.shape(self.timeArray), dtype=float)

        for i, temperature_i in enumerate(self.steelTemperatureArray):
            # calculate strength
            a, b, c = \
                fd_eurocode.reduction_factor_carbon_steel(steel_temperature = temperature_i + 273.15)
            k_y_theta[i] = a
            k_p_theta[i] = b
            k_E_theta[i] = c
            lamda_theta[i] = fd_eurocode.reduction_factor_buckling_fire_carbon_steel(
                stress_yield=stress_yield,lambda_=self.slenderness,k_y_theta=k_y_theta[i],k_E_theta=k_E_theta[i])

            # calculate thermal expansion
            strain_theta[i] = fd_eurocode.relative_thermal_elongation_carbon_steel(steel_temperature= temperature_i + 273.5)
            # calculate thermal stress
            stress_theta[i] = fd_eurocode.stress_strain_elevated_temperature_carbon_steel(
                steel_strain=strain_theta[i],
                stress_proportional=k_p_theta[i] * self.steelProportionalStress,
                stress_yield=self.steelYieldStress,
                elastic_modulus=k_E_theta[i] * self.steelElasticModulus)

        # return reduction factors
        self.steelEffectiveYieldReductionFactor = k_y_theta
        self.steelProportionalLimitReductionFactor = k_p_theta
        self.steelElasticModulusReductionFactor = k_E_theta

        # return strength variation
        gamma_m_fi = 1.0  # from national annex, recommended as 1.0 for fire
        self.steelStrength = lamda_theta * self.steelYieldStress * k_y_theta / gamma_m_fi
        self.steelSlendernessTemperatureDependent = lamda_theta

        # return thermal stress
        self.steelThermalStrain = strain_theta
        self.steelThermalStress = stress_theta
