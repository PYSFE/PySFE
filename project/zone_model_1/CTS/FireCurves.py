# -*- coding: utf-8 -*-
import numpy as np

class ISO834(object):
    """
        Description:
            generates so called the 'standard fire curve' based on ISO 834
        Attributs:
            timeStep            {double, s⁻¹}           time step between each interval
            timeLength          {double, s}             time span, i.e. x-axis limit
            ambientTemperature  {double, ℃}             ambient temperature
        Methods:
            TimeInMinutes()     {ndarray, min}          convert time array data into minutes
    """
    def __init__(self, timeStep = 1.0 , timeLength = 18000, ambientTemperature = 25):
        self.timeStep = timeStep
        self.timeLength = timeLength
        self.ambientTemperature = ambientTemperature
        
        # START: time-temperature curve calculation
        timeArray = np.arange(0, timeLength + timeStep, timeStep)
        temperatureArray = 345. * np.log10(timeArray[1:]/60. * 8. + 1.) + ambientTemperature
        temperatureArray = np.insert(temperatureArray, 0, ambientTemperature)
        # END: time-temperature curve calculation

        self.timeArray = timeArray
        self.temperatureArray = temperatureArray

    @property
    def ArraySize(self):
        return np.size(self.timeArray)


class ASTM_E119(object):
    """
        Description:
            generates fire curve according to ASTM_E119.
        Attributs:
            timeStep            {double, s⁻¹}           time step between each interval.
            timeLength          {double, s}             time span, i.e. x-axis limit.
            ambientTemperature  {double, ℃}             ambient temperature.
        Methods:
            TimeInMinutes()     {ndarray, min}          convert time array data into minutes.
    """
    def __init__(self, timeStep = 1.0, timeLength = 18000, ambientTemperature = 25):
        self.timeStep = timeStep
        self.timeLength = timeLength
        self.ambientTemperature = ambientTemperature

        # START: time-temperature curve calculation
        timeArray = np.arange(timeStep,timeLength,timeStep)
        timeArray = timeArray/60/60 # convert from seconds to hours
        temperatureArray = 750 * (1 - np.exp(-3.79553 * np.sqrt(timeArray))) + 170.41 * np.sqrt(timeArray) + ambientTemperature
        timeArray = np.insert(timeArray * 60 * 60,0,0) # convert time array from hours to seconds, insert initial condition
        temperatureArray = np.insert(temperatureArray, ambientTemperature, 0) # insert initial condition
        # END: time-temperature curve calculation

        self.timeArray = timeArray
        self.temperatureArray = temperatureArray


class Eurocode_Hydrocarbon(object):
    """
        Description:
            generates Eurocode hydrocarbon time-temperature fire curve in numpy array
        Attributs:
            timeStep            {double, s⁻¹}           time step between each interval
            timeLength          {double, s}             time span, i.e. x-axis limit
            ambientTemperature  {double, ℃}             ambient temperature
        Methods:
            TimeInMinutes()     {ndarray, min}          convert time array data into minutes
    """
    def __init__(self, timeStep = 1.0, timeLength = 18000, ambientTemperature = 25):
        self.timeStep = timeStep
        self.timeLength = timeLength
        self.ambientTemperature = ambientTemperature

        # START: time-temperature curve calculation
        timeArray = np.arange(timeStep,timeLength,timeStep)
        timeArray = timeArray / 60 # convert from seconds to hours
        temperatureArray = 1080 * (1 - 0.325 * np.exp(-0.167 * timeArray) - 0.675 * np.exp(-2.5 * timeArray)) + ambientTemperature
        timeArray = np.insert(timeArray * 60,0,0) # convert time array from hours to seconds, insert initial condition
        temperatureArray = np.insert(temperatureArray,ambientTemperature,0) # insert initial condition
        # END: time-temperature curve calculation

        self.timeArray = timeArray
        self.temperatureArray = temperatureArray

    def TimeInMinutes(self):
        return (self.timeArray/60.0)
    

class Eurocode_ExternalFire(object):
    """
        Description:
            generates Eurocode external fire time-temperature curve
        Attributs:
            timeStep            {double, s⁻¹}           time step between each interval
            timeLength          {double, s}             time span, i.e. x-axis limit
            ambientTemperature  {double, ℃}             ambient temperature
        Methods:
            TimeInMinutes()     {ndarray, min}          convert time array data into minutes
    """
    def __init__(self, timeStep = 1.0, timeLength = 18000, ambientTemperature = 25):
        self.timeStep = timeStep
        self.timeLength = timeLength
        self.ambientTemperature = ambientTemperature

        # START: time-temperature curve calculation
        timeArray = np.arange(timeStep,timeLength,timeStep)
        timeArray = timeArray / 60. # convert from seconds to hours
        temperatureArray = 660 * (1 - 0.687 * np.exp(-0.32 * timeArray) - 0.313 * np.exp(-3.8 * timeArray)) + ambientTemperature
        timeArray = np.insert(timeArray * 60,0,0) # convert time array from hours to seconds, insert initial condition
        temperatureArray = np.insert(temperatureArray,ambientTemperature,0) # insert initial condition
        # END: time-temperature curve calculation

        self.timeArray = timeArray
        self.temperatureArray = temperatureArray


class EurocodeParametricFire(object):
    """
    Description:
        BS EN 1991-1-2:2002, Annex A
        Generates Eurocode external fire time-temperature curve
    Attributes:
        totalEnclosureArea      {float, m²}             Total internal surface area, including openings
        floorArea               {float, m²}             Floor area
        openingArea             {float, m²}             Vertical open area
        weightedOpeningHeight   {float, m}              Weighted average vertical openings height
        density                 {float, kg/m³}          Density of enclosure
        specificHeat            {float, J/kg/K}         Specific heat of enclosure
        thermalConductivity     {float, W/m/K}          Thermal conductivity of enclosure
        fireLoadDensityFloor    {float, MJ/m²}          Fuel load on floor area
        fireGrowthRate          {string, -}             Can be "low", "medium" or "fast"

        timeStep                {float, s⁻¹}            Time step between each interval
        timeLengthCap           {float, s}              Maximum allowable time
    Methods:
        TimeInMinutes()         {ndarray, min}          convert time array data into minutes
    Remarks:
        *   When estimating extinction time, a upper bound value of 12 hours is used for approximation. Code will
            fail to find the extinguish time if the fire does not extinguish within 12 hours. A maximum 1000 loops
            is constrained.
        *   0.02 <= opening factor <= 0.002
        *   100 <= b (thermal inertia) <= 2200
    """
    def __init__(self,
                 total_enclosure_area, floor_area, opening_area, opening_height,
                 density, specific_heat, thermal_conductivity,
                 fire_load_density_floor, fire_growth_rate,
                 time_step=0.1, time_extend_after_extinction=1800, is_proceed=True):

        # INPUT PROPERTIES
        self.timeStep = time_step / 3600.
        self.timeAfterExtinction = time_extend_after_extinction / 3600.
        self.totalArea = float(total_enclosure_area)
        self.floorArea = float(floor_area)
        self.openingArea = float(opening_area)
        self.openingHeight = float(opening_height)
        self.fireLoadDensityFloor = float(fire_load_density_floor)
        self.fireGrowthRate = fire_growth_rate

        # DERIVED PROPERTIES
        fireGrowthRateContainer = {"slow": 25.0 / 60.0, "medium": 20.0 / 60.0, "fast": 15.0 / 60.0}  # (10)
        self.openingFactor = self.openingArea * np.sqrt(self.openingHeight) / self.totalArea
        self.b = np.sqrt(density * specific_heat * thermal_conductivity)
        self.lamda = np.power(self.openingFactor / self.b, 2) / np.power(0.04 / 1160, 2)
        self.fireLoadDensityTotal = self.fireLoadDensityFloor * (self.floorArea / self.totalArea)
        self.timeLimiting = fireGrowthRateContainer[fire_growth_rate]  # (10)
        self.openingFactorLimiting = 0.1e-3 * self.fireLoadDensityTotal / self.timeLimiting  # A.9
        self.lamdaLimiting = np.power((self.openingFactorLimiting / self.b), 2) / np.power(0.04 / 1160, 2)
        if self.openingFactor>0.04 and self.fireLoadDensityTotal < 75 and self.b < 1160:
            # (A.10)
            self.lamdaLimiting *= (1 +
                                   ((self.openingFactor - 0.04) / 0.04) *
                                   ((self.fireLoadDensityTotal - 75) / 75) *
                                   ((1160 - self.b) / 1160))
        self.timePeakTemperature = max([0.2e-3 * self.fireLoadDensityTotal / self.openingFactor, self.timeLimiting])

        # OUTPUT PROPERTIES (CONTAINER)
        self.peakTemperature = None
        self.timeArray = None
        self.temperatureArray = None

        if is_proceed:
            self.proceed()

    """
    DESCRIPTION: Estimates gas temperature during heating phase, requires self.peakTemperature
    """
    def heating_phase_gas_temperature(self, timeArrayHeating):
        if self.timePeakTemperature == self.timeLimiting:
            # (A.2b)
            tt = timeArrayHeating * self.lamdaLimiting
        else:
            # (A.2a)
            tt = timeArrayHeating * self.lamda

        return 20 + 1325 * (1 - 0.324 * np.exp(-0.2*tt) - 0.204 * np.exp(-1.7*tt) - 0.472 * np.exp(-19*tt))

    def cooling_phase_gas_temperature_extinction_time(self):
        ttMax = (
                0.2e-3 * self.fireLoadDensityTotal / self.openingFactor) * self.lamda  # (A.12) explicitly for cooling phase
        if (self.timePeakTemperature > self.timeLimiting):  # (A.12), explicitly for cooling phase
            x = 1.0
        elif (self.timePeakTemperature == self.timeLimiting):
            x = self.timeLimiting * self.lamda / ttMax
        gasTemperature = self.peakTemperature

        upperBound = 12 * 3600
        lowerBound = self.timePeakTemperature
        t = None

        loopCount = 0
        while (not(19.5 < gasTemperature < 20.5) and loopCount <= 10000):
            if t == None:
                t = 0.5 * (upperBound + lowerBound)
            else:
                if gasTemperature > 20:
                    lowerBound = t
                    t = 0.5 * (lowerBound + upperBound)
                elif gasTemperature < 20:
                    upperBound = t
                    t = 0.5 * (lowerBound + upperBound)
            if (ttMax <= 0.5): # (A.11a)
                gasTemperature = self.peakTemperature - 625 * (t * self.lamda - ttMax * x)
            elif (0.5 < ttMax < 2): # (A.11a)
                gasTemperature = self.peakTemperature - 250 * (3 - ttMax) * (t * self.lamda - ttMax * x)
            elif (ttMax >= 2): # (A.11a)
                gasTemperature = self.peakTemperature - 250 * (t * self.lamda - ttMax * x)
            loopCount += 1

        return t

    def cooling_phase_gas_temperature(self, time_array_cooling):
        tt = time_array_cooling * self.lamda  # (A.12), explicitly for cooling phase
        ttMax = (0.2e-3 * self.fireLoadDensityTotal / self.openingFactor) * self.lamda # (A.12) explicitly for cooling phase
        if self.timePeakTemperature > self.timeLimiting:  # (A.12), explicitly for cooling phase
            x = 1.0
        elif self.timePeakTemperature == self.timeLimiting:
            x = self.timeLimiting * self.lamda / ttMax
        else:
            x = None

        if ttMax <= 0.5:  # (A.11a)
            gas_temperature = self.peakTemperature - 625 * (tt - ttMax * x)
        elif 0.5 < ttMax < 2:  # (A.11a)
            gas_temperature = self.peakTemperature - 250 * (3 - ttMax) * (tt - ttMax * x)
        elif ttMax >= 2:  # (A.11a)
            gas_temperature = self.peakTemperature - 250 * (tt - ttMax * x)
        else:
            gas_temperature = np.zeros(np.shape(tt))

        return gas_temperature

    def proceed(self):
        # HEATING PHASE
        time_array_heating = np.arange(0, int(self.timePeakTemperature/self.timeStep)*self.timeStep+self.timeStep, self.timeStep)
        temperature_array_heating = self.heating_phase_gas_temperature(time_array_heating)
        # self.peakTemperature = temperatureArrayHeating.max()
        self.peakTemperature = max(temperature_array_heating)

        # COOLING PHASE
        extinguish_time = self.cooling_phase_gas_temperature_extinction_time()
        time_array_cooling = np.arange(time_array_heating.max()+self.timeStep, int(extinguish_time/self.timeStep)*self.timeStep+self.timeStep, self.timeStep)
        temperature_array_cooling = self.cooling_phase_gas_temperature(time_array_cooling)
        temperature_array_cooling[temperature_array_cooling < 25] = 25

        # EXTEND PHASE
        timeArrayExtend = np.arange(time_array_cooling.max()+self.timeStep, time_array_cooling.max()+self.timeStep+self.timeAfterExtinction, self.timeStep)
        temperatureArrayExtend = np.full((np.size(timeArrayExtend),), 25)

        # Assemble array
        self.timeArray = np.concatenate((time_array_heating,time_array_cooling,timeArrayExtend))
        self.temperatureArray = np.concatenate((temperature_array_heating, temperature_array_cooling, temperatureArrayExtend))

        # Convert time to seconds
        self.timeArray *= 3600
        self.timeStep *= 3600
