# -*- coding: utf-8 -*-
# Calc Version: 1.00

import numpy as np
from project.zone_model_1.CTS import CTS
from project.zone_model_1.CTS import FireCurves
from project.zone_model_1.CTS import FireDynamics
from project.zone_model_1.CTS.FireDynamics import gaseous_chamber_ambient_pressure
from scipy.interpolate import interp1d


def main_function(
        time_step,
        time_cap,
        ambient_air_density,
        ambient_air_pressure,
        ambient_temperature,
        gas_emissivity,
        ventilation_modifier_fixed,
        enclosure_volume,
        enclosure_gas_pressure,
        fuel_material,
        specimen_surface_emissivity,
        specimen_exposed_area,
        specimen_thickness,
        specimen_heat_of_combustion,
        specimen_density,
        specimen_moisture_content,  # 0.12 ONLY, only have rho data with w=0.12 softwood
        lining_surface_emissivity,
        lining_surface_area,
        lining_thickness,
        lining_conductivity,
        lining_density,
        lining_specific_heat,
        window_area,
        **kwargs
):
    """
        MODEL ASSUMPTIONS:
            1. Energy loss to opening is ignored as no opening exists
            2. Energy loss to specimen is ignored
            3. Energy loss to internal air is ignored
            4. Specimen combustion is assumed to be immediate after disperse
            5. Specimen burning rate (charring rate for timber) is set to constant 0.7 [mm/min] by default
            6. Ventilation mass flow rate is set to constant 0.0735 [kg/s] by default
        FURTHER PLANNING:
            1. timber heat transfer model
            2. timber combustion model
    """
    # ============================================= preliminary variables ==============================================
    # ***** Setting
    status = "Success"
    warning_max_temperature = 1200. + 273.15  # [K], warning message when temperature_gas is greater than this figure.
    temperature_deviation_increment = 0.25  # [K], if hr_burner become negative, temperature_gas will be increased by this figure.
    temperature_ignation_specimen = 390. + 273.15  # data from largish specimen exposed to radiant, open enviornment

    specimen_material = "cellulose"
    lining_virtual_layer_thickness = 0.005  # [m]
    specimen_virtual_layer_thickness = 0.005  # [m]
    dict_material_to_formula = {"propane": "C3H8", "methane": "CH4"}
    fuel_material_formula = dict_material_to_formula[fuel_material]
    gas_species_list = ("N2", "O2", "CO2", "Ar", "H2O", fuel_material_formula)

    # ***** Physical Properties
    ideal_gas = 8.3144598  # ideal gas constant, R, [J / K / mol]
    stefan_boltzmann = 5.67e-8  # Stefan-Boltzmann constant, sigma, [W / m2 / K4]
    gases_molar_mass = {  # [kg/mol]
        "O2": 31.9988 / 1000.,
        "CO2": 44.0100 / 1000.,
        "N2": 28.0134 / 1000.,
        "H2O": 18.0200 / 1000.,
        "Ar": 39.9480 / 1000.,
        "C3H8": 44.0956 / 1000.,
        "CH4": 16.0425 / 1000.,
    }
    k_w = FireDynamics.ThermalConductivity(lining_conductivity.replace("_", " "))  # thermal conductivity of walls, ceiling and floors
    k_s = FireDynamics.ThermalConductivity("timber ec5-1-2")
    rho_ratio_s = FireDynamics.DensityRatio("timber ec5-1-2")
    c_s = FireDynamics.SpecificHeatP("timber ec5-1-2")

    # ***** Ambient conditions
    content_mole_ambient_air = {"O2": 0.209500, "N2": 0.780870, "CO2": 0.000300, "Ar": 0.009330, "H2O": 0.000000}
    content_mass_ambient_air = {"O2": 0.232000, "N2": 0.755154, "CO2": 0.000046, "Ar": 0.012800, "H2O": 0.000000}
    mass_initial_ventilation_in = ventilation_modifier_fixed  # [kg/s]

    # ***** Drived Variables
    emission_coefficient = 1.1  # K [m-1], obtained from Drysdale
    flame_thickness = np.sqrt(lining_surface_area/6)  # x_F [m], assumed to be the average value of depth, width & height
    gas_effective_emissivity = 1 - np.exp(-emission_coefficient * flame_thickness)  # e_F, equation from Drysdale

    e_r_lining = 1 / (1 / gas_emissivity + 1 / lining_surface_emissivity - 1)  # resultant emissivity, lining
    e_r_specimen = 1 / (1 / gas_emissivity + 1 / specimen_surface_emissivity - 1)  # resultant emissivity, specimen

    initial_furnace_o2_mass = (ambient_air_pressure * enclosure_volume / ideal_gas / ambient_temperature * gases_molar_mass["O2"]) * content_mass_ambient_air["O2"]
    initial_ventilation_o2_mass = mass_initial_ventilation_in * content_mass_ambient_air["O2"]

    ventilation_modifier_fixed *= content_mass_ambient_air["O2"]  # convert from air to O2

    # define lining material properties for calculating temperature profile (finite difference)
    lining_slices = int(lining_thickness/lining_virtual_layer_thickness+1)
    lining_container_arr = np.ones(shape=(lining_slices,), dtype=float)
    lining_thickness_arr = lining_container_arr * (lining_thickness/lining_slices)
    lining_density_arr = lining_container_arr * lining_density
    lining_thermal_conductivity_arr = lining_container_arr * k_w.temp(800+273.15)
    lining_specific_heat_arr = lining_container_arr * lining_specific_heat

    # define specimen material properties for calculating temperature profile (finite difference)
    specimen_slices = int(specimen_thickness/specimen_virtual_layer_thickness+1)
    specimen_container_arr = np.ones(shape=(specimen_slices,), dtype=float)
    specimen_thickness_arr = specimen_container_arr * (specimen_thickness/specimen_slices)
    specimen_density_arr = specimen_container_arr * specimen_density
    specimen_thermal_conductivity_arr = specimen_container_arr * k_s.temp(800+273.15)
    specimen_specific_heat_arr = specimen_container_arr * c_s.temp(800+273.15)

    # get gas temperature curve
    fire_curve = FireCurves.ISO834(time_step, time_cap, ambient_temperature - 273.15)
    temperature_gas_target = np.asarray(fire_curve.temperatureArray) + 273.15  # FireCurve.ISO834 is in degree Celsius

    # create output container
    container_shape = np.shape(temperature_gas_target)
    gas_in0 = gaseous_chamber_ambient_pressure(gases_molar_mass, container_shape[0])
    gas_in1 = gaseous_chamber_ambient_pressure(gases_molar_mass, container_shape[0])
    gas_in2 = gaseous_chamber_ambient_pressure(gases_molar_mass, container_shape[0])
    gas_en0 = gaseous_chamber_ambient_pressure(gases_molar_mass, container_shape[0])
    gas_en1 = gaseous_chamber_ambient_pressure(gases_molar_mass, container_shape[0])
    gas_en2 = gaseous_chamber_ambient_pressure(gases_molar_mass, container_shape[0])
    gas_ou = gaseous_chamber_ambient_pressure(gases_molar_mass, container_shape[0])
    content_vol_species_en = {gas: np.zeros(container_shape, float) for gas in gases_molar_mass}
    c_species_ou = {gas: np.zeros(container_shape, float) for gas in gases_molar_mass}
    temperature_lining = np.zeros(shape=(len(temperature_gas_target), lining_slices))
    temperature_specimen = np.zeros(shape=(len(temperature_gas_target), specimen_slices))
    temperature_window = np.zeros(shape=(len(temperature_gas_target), lining_slices))
    temperature_gas = np.zeros(container_shape, float)  # [K], actual internal gas temperature

    hr_lining = np.zeros(container_shape, float)  # HR exchange due to lining mass
    hr_ventilation = np.zeros(container_shape, float)  # HR exchange due to ventilation mass
    hr_gas_enclosure = np.zeros(container_shape, float)  # HR exchange due to enclosure internal gas mass
    hr_specimen_loss = np.zeros(container_shape, float)  # HR exchange due to the specimen (timber) mass
    hr_specimen_combustion = np.zeros(container_shape, float)  # HR release from the specimen (timber)
    hr_burner = np.ones(container_shape, float) * -1.  # total energy gain from the burner, negative for while condition
    hr_windows = np.zeros(container_shape, float)  # HR due to windows via radiation
    hr_residual = np.zeros(container_shape, float)  # HR that when hr_burner was forced to set as 0.00
    hr_windows_conduction = np.zeros(container_shape, float)  # HR due to windows mass
    # Since burner hr can not be negative, 'hr_residual' is the hr required to cool the furnace down to targeted temperature

    mass_fuel_in = np.zeros(container_shape, float)  # fuel burning rate required to heat up the furnace
    mass_specimen_in = np.zeros(container_shape, float)  # specimen (timber) burning rate, this is probably given
    length_specimen_charring = np.zeros(container_shape, float)  # specimen charring rate, used for calc specimen burning
    hf_furnace_average = np.zeros(container_shape, float)  # average heat flux within the furnace
    hf_furnace = np.zeros(container_shape, float)  # instantaneous heat flux within the furnace

    mass_o2_specimen_required = np.zeros(container_shape, float)
    mass_co2_specimen_produced = np.zeros(container_shape, float)
    mass_h2o_specimen_produced = np.zeros(container_shape, float)
    mass_o2_burner_required = np.zeros(container_shape, float)
    mass_co2_burner_produced = np.zeros(container_shape, float)
    mass_h2o_burner_produced = np.zeros(container_shape, float)

    # assign values for initial condition, conditions which is zero are not declared again
    temperature_lining[0, :] = ambient_temperature
    temperature_specimen[0, :] = ambient_temperature
    temperature_gas[0] = ambient_temperature

    gas_in0.set_mass_by_proportion("O2", initial_ventilation_o2_mass, content_mass_ambient_air, 0)
    gas_in1.set_mass_by_proportion("O2", initial_ventilation_o2_mass, content_mass_ambient_air, 0)
    gas_in2.set_mass_by_proportion("O2", initial_ventilation_o2_mass, content_mass_ambient_air, 0)
    gas_en0.set_mass_by_proportion("O2", initial_furnace_o2_mass, content_mass_ambient_air, 0)
    gas_en1.set_mass_by_proportion("O2", initial_furnace_o2_mass, content_mass_ambient_air, 0)
    gas_en2.set_mass_by_proportion("O2", initial_furnace_o2_mass, content_mass_ambient_air, 0)
    gas_ou.set_mass_by_proportion("O2", initial_ventilation_o2_mass, content_mass_ambient_air, 0)

    # ============================================ time evolution starts ===============================================
    for i in np.arange(1, len(temperature_gas_target)):
        count_temperature_deviation_increase = -1
        temperature_gas[i] = np.max([temperature_gas_target[i], temperature_gas[i-1]]) - temperature_deviation_increment
        while hr_burner[i] < 0:
            count_temperature_deviation_increase += 1
            # Step 0.1: Get the target temperature, ISO 834 ------------------------------------------------------------
            temperature_gas[i] += temperature_deviation_increment
            hf_furnace_average[i] = np.average(heat_flux_furnace_babrauskas2005(temperature_gas[:(i+1)]))

            # Step 0.2: Calculate thermal properties -------------------------------------------------------------------
            lining_thermal_conductivity_arr = np.asarray([k_w.temp(v) for v in temperature_lining[i-1, :]])

            # Step 1: Energy associated with lining --------------------------------------------------------------------
            # Estimate energy loss to lining, including convective and radiation heat transfer. This is only depend on the
            # temperature difference between inside and outside the furnace at the currently time step.
            inner_temperature = temperature_gas[i]
            inner_conductivity = e_r_lining * stefan_boltzmann * (inner_temperature ** 4 - temperature_lining[i-1, 0] ** 4) / abs(inner_temperature - temperature_lining[i-1, 0]) + 23.
            outer_temperature = ambient_temperature
            outer_conductivity = 0.033 * outer_temperature - 0.309
            # Calculate lining temperature profile.
            dT_dt = heat_transfer_general_1d_finite_difference(
                lining_thermal_conductivity_arr,
                lining_density_arr,
                lining_specific_heat_arr,
                temperature_lining[i-1, :],
                lining_thickness_arr,
                time_step, inner_conductivity, inner_temperature, outer_conductivity, outer_temperature,
            )
            dT = dT_dt * time_step
            temperature_lining[i, :] = temperature_lining[i-1, :] + dT
            # Calculate energy loss rate to lining.
            hr_lining[i] = lining_surface_area * (inner_temperature - temperature_lining[i, 0]) / (1 / inner_conductivity + 0.5 * lining_thickness_arr[0] / lining_thermal_conductivity_arr[0])
            hr_lining[i] = - hr_lining[i]

            # Step 2: Energy associated with windows -------------------------------------------------------------------
            # Estimate energy loss to windows, including radiative and conductive heat transfer.
            # energy loss rate to windows (radiation)
            hr_windows[i] = window_area * gas_effective_emissivity * 5.67e-8 * (temperature_gas[i]**4 - ambient_temperature**4)
            hr_windows[i] = - hr_windows[i]

            # Step 4: Energy associated with the specimen --------------------------------------------------------------
            # Timber combustion. Estimate heat release, cellulose, O2, CO2 and H2O mass rates due to timber combustion.
            if specimen_exposed_area > 0.:
                specimen_thermal_conductivity_arr = np.asarray([k_s.temp(v) for v in temperature_specimen[i - 1, :]])
                specimen_density_arr = np.asarray(
                    [rho_ratio_s.temp(v) * specimen_density for v in temperature_specimen[i - 1, :]])
                specimen_specific_heat_arr = np.asarray([c_s.temp(v) for v in temperature_specimen[i - 1, :]])

                length_specimen_charring[i] = timber_charring_rate_babrauskas2005(
                    average_heat_flux=hf_furnace_average[i],
                    timber_density=specimen_density,
                    exposure_time=i*time_step,
                    oxygen_content_volume=gas_en1.get_content_mole("O2", i-1)
                )
                # length_specimen_charring[i] = timber_charring_rate(specimen_thickness, i)  # time dependent charring rate
                if temperature_specimen[i-1, 0] > temperature_ignation_specimen:
                    mass_specimen_in[i] = length_specimen_charring[i] * specimen_exposed_area * specimen_density
                else:
                    mass_specimen_in[i] = 0

                # heat release is estimated from charring rate and oxygen avaliable, which ever is smaller
                hr_specimen_combustion_charring = mass_specimen_in[i] * specimen_heat_of_combustion
                hr_specimen_combustion_oxygen = (gas_en1.get_mass("O2", i-1)/float(time_step)) / combustion_chemistry(1., 'cellulose', 'O2 required')
                # ([kg] of O2 avaliable) / ([kg/J] of O2 required per J of energy release from timber)
                hr_specimen_combustion[i] = np.min([hr_specimen_combustion_charring, hr_specimen_combustion_oxygen])
                mass_o2_specimen_required[i] = combustion_chemistry(hr_specimen_combustion[i], specimen_material, 'O2 required')
                mass_co2_specimen_produced[i] = combustion_chemistry(hr_specimen_combustion[i], specimen_material, 'CO2 produced')
                mass_h2o_specimen_produced[i] = combustion_chemistry(hr_specimen_combustion[i], specimen_material, 'H2O produced')

                # Heat loss rate to specimen
                # Estimate energy loss to specimen, including convective and radiation heat transfer. This is depend on the
                # temperature difference between inside and outside the furnace at the currently time step.
                inner_temperature = temperature_gas[i]
                inner_conductivity = e_r_specimen * stefan_boltzmann * (inner_temperature ** 4 - temperature_specimen[i-1, 0] ** 4) / abs(inner_temperature - temperature_specimen[i-1, 0]) + 23.
                outer_temperature = ambient_temperature
                outer_conductivity = 0.033 * outer_temperature - 0.309
                # Calculate specimen temperature profile.
                dT_dt = heat_transfer_general_1d_finite_difference(
                    specimen_thermal_conductivity_arr,
                    specimen_density_arr,
                    specimen_specific_heat_arr,
                    temperature_specimen[i-1, :],
                    specimen_thickness_arr,
                    time_step, inner_conductivity, inner_temperature, outer_conductivity, outer_temperature,
                )
                dT = dT_dt * time_step
                temperature_specimen[i, :] = temperature_specimen[i-1, :] + dT
                if temperature_specimen[i, 0] > temperature_gas[i]:
                    print("Specimen temperature > gas temperature at " + str(i))
                # Calculate energy loss rate to lining.
                hr_specimen_loss[i] = specimen_exposed_area * (inner_temperature - temperature_specimen[i, 0]) / (1 / inner_conductivity + 0.5 * specimen_thickness_arr[0] / specimen_thermal_conductivity_arr[0])
                hr_specimen_loss[i] = - hr_specimen_loss[i]

            # Step 5: Energy associated with gas exchange --------------------------------------------------------------
            energy_gas_en = gas_en2.calc_energy_for_temperature_raise(temperature_gas[i-1], temperature_gas[i], i-1) / time_step
            energy_gas_ou = gas_ou.calc_energy_for_temperature_raise(ambient_temperature, temperature_gas[i], i-1)
            hr_gas_enclosure[i] = - energy_gas_en
            hr_ventilation[i] = - energy_gas_ou

            # Step 6: Energy associated with the burner ----------------------------------------------------------------
            # Burner combustion. Estimate heat release, fuel (propane/natural gas), O2, CO2 and H2O mass rates due to fuel
            # combustion.
            hr_burner[i] = - hr_lining[i]\
                           - hr_ventilation[i]\
                           - hr_gas_enclosure[i]\
                           - hr_windows[i]\
                           - hr_specimen_combustion[i]\
                           - hr_specimen_loss[i]
        if count_temperature_deviation_increase > 0:
            hr_burner[i] = 0.
        mass_fuel_in[i] = combustion_chemistry(hr_burner[i], fuel_material, 'fuel required')
        mass_o2_burner_required[i] = combustion_chemistry(hr_burner[i], fuel_material, 'O2 required')
        mass_co2_burner_produced[i] = combustion_chemistry(hr_burner[i], fuel_material, 'CO2 produced')
        mass_h2o_burner_produced[i] = combustion_chemistry(hr_burner[i], fuel_material, 'H2O produced')

        # Step 7: Gases ------------------------------------------------------------------------------------------------
        # ***** Ventilation in based on O2 requirement
        # before combustion
        gas_in0.set_mass_by_proportion("O2", mass_o2_burner_required[i] + ventilation_modifier_fixed, content_mass_ambient_air, i)  # set mass for air in
        gas_in0.set_mass(fuel_material_formula, mass_fuel_in[i], i)  # set mass for fuel
        # set mass for tars ?

        dict_gas_1 = {
            "O2": -mass_o2_burner_required[i],
            "CO2": mass_co2_burner_produced[i],
            "H2O": mass_h2o_burner_produced[i],
            fuel_material_formula: -mass_fuel_in[i],
        }
        dict_gas_2 = {
            "O2": - mass_o2_burner_required[i] - mass_o2_specimen_required[i],
            "CO2": mass_co2_burner_produced[i] + mass_co2_specimen_produced[i],
            "H2O": mass_h2o_burner_produced[i] + mass_h2o_specimen_produced[i],
            fuel_material_formula: -mass_fuel_in[i],
        }

        for key in gas_species_list:
            # in
            m0 = gas_in0.get_mass(key, i)
            dm1 = dict_gas_1[key] if key in dict_gas_1 else 0.
            dm2 = dict_gas_2[key] if key in dict_gas_2 else 0.
            gas_in1.set_mass(key, m0 + dm1, i)  # burner combustion
            gas_in2.set_mass(key, m0 + dm2, i)  # burner + specimen combustion

            # mixing
            dm0 = gas_in0.get_mass(key, i) * time_step
            dm1 = gas_in1.get_mass(key, i) * time_step
            dm2 = gas_in2.get_mass(key, i) * time_step
            gas_en0.set_mass(key, gas_en2.get_mass(key, i-1) + dm0, i)  # mix with fresh air
            gas_en1.set_mass(key, gas_en2.get_mass(key, i-1) + dm1, i)  # mix with mod 1
            gas_en2.set_mass(key, gas_en2.get_mass(key, i-1) + dm2, i)  # mix with mod 2

        # out
        n_existing = gas_en2.get_mole_total(i)  # no. of moles within the furnace
        n_permissible = (enclosure_gas_pressure * enclosure_volume / ideal_gas / temperature_gas[i]) / n_existing
        n_exceeded = 1 - n_permissible
        for key in gas_species_list:
            gas_ou.set_mass(key, gas_en2.get_mass(key, i) * n_exceeded / time_step, i)
            gas_en2.set_mass(key, gas_en2.get_mass(key, i) * n_permissible, i)

        debug_print_header = ["time", "furnace temperature", "gas temperature dev.", "oxygen content", "fuel in", "specimen in", "gas in", "gas out", "specimen heat rate", "burner heat rate"]
        debug_print_value = [i*time_step, temperature_gas[i], temperature_gas[i]-temperature_gas_target[i], gas_en1.get_content_mole("O2", i)*100, mass_fuel_in[i], mass_specimen_in[i], gas_in2.get_mass_total(i), gas_ou.get_mass_total(i), hr_specimen_combustion[i]/1.e3, hr_burner[i]/1.e3]
        debug_print_suffix = ["[s]", "[K]", "[K]", "[%]", "[kg]", "[kg]", "[kg]", "[kg]", "[kJ]", "[kJ]"]
        print_debug_information("{:20.20}: {:07.2f} {}\n", zip(debug_print_header, debug_print_value, debug_print_suffix), "-"*35)

        # error check
        if temperature_gas[i] > warning_max_temperature:
            status = "Failed: Maximum temperature {} [K] has reached".format(round(temperature_gas[i],1))
            break
        elif (np.max(temperature_specimen[i,:])>temperature_gas[i] or np.min(temperature_specimen[i,:])<ambient_temperature) and specimen_exposed_area > 0.:
            status = "Failed: Specimen temperature out of range"
            break
        elif (np.max(temperature_specimen[i,:])>temperature_gas[i] or np.min(temperature_specimen[i,:])<ambient_temperature) and specimen_exposed_area > 0.:
            status = "Failed: Lining temperature out of range"
            break

    # post process lining temperature profile data
    temperature_lining = np.transpose(temperature_lining)
    temperature_lining.tolist()
    temperature_specimen = np.transpose(temperature_specimen)
    temperature_specimen.tolist()
    # calculate heat flux from temperature
    hf_furnace = heat_flux_furnace_babrauskas2005(temperature_ndarray=temperature_gas)

    # create output data
    data_dict = {
        "time": fire_curve.timeArray,

        "temperature_gas": temperature_gas,
        "temperature_gas_target": temperature_gas_target,
        "temperature_lining": temperature_lining,
        "temperature_specimen": temperature_specimen,

        "hr_ventilation": hr_ventilation,
        "hr_gas_enclosure": hr_gas_enclosure,
        "hr_lining": hr_lining,
        "hr_windows": hr_windows,
        "hr_specimen_combustion": hr_specimen_combustion,
        "hr_specimen": hr_specimen_loss,
        "hr_burner_combustion": hr_burner,

        "length_specimen_charring": length_specimen_charring,
        "hf_furnace_average": hf_furnace_average,
        "hf_furnace": hf_furnace,

        "mass_fuel_in": mass_fuel_in,
        "mass_o2_burner_required": mass_o2_burner_required,
        "mass_co2_burner_produced": mass_co2_burner_produced,
        "mass_h2o_burner_produced": mass_h2o_burner_produced,
        "mass_specimen_in": mass_specimen_in,
        "mass_o2_specimen_required": mass_o2_specimen_required,
        "mass_co2_specimen_produced": mass_co2_specimen_produced,
        "mass_h2o_specimen_produced": mass_h2o_specimen_produced,
        "mass_ventilation_in": gas_in0.get_mass_total(),
        "mass_ventilation_ou": gas_ou.get_mass_total(),
    }
    for key, val in gas_in0.get_mass().iteritems():
        data_dict["mass_"+key] = val
    for key, val in gas_ou._gases_cp.iteritems():
        data_dict["c_"+key] = val
    for key, val in gas_en1.get_content_mole().iteritems():
        data_dict["content_vol_"+key] = val

    return data_dict, status


def timber_charring_rate(specimen_thickness, t_arr):  # unit: [m/s]
    with np.errstate(divide='ignore', invalid = 'ignore'):
        charring_rate = 1.30 / ((t_arr / 60.) ** 0.2)  # [mm/min]
        charring_rate /= 1000  # [m/min]
        charring_rate /= 60  # [m/s]

    return charring_rate


# heat flux in respect to temperature
# based on heat q = f(t) from Babrauskas 2005 and T = f(t) from ISO 834
def heat_flux_furnace_babrauskas2005(temperature_ndarray):
    T = temperature_ndarray
    t = (np.power((T-293.15)/345., 10.)) / 8.
    heat_flux = 25200. * np.power(t, 0.4)
    return heat_flux


# DESCRIPTION:
# This timber charring rate is from Babrauskas 2005 - Charring rate of wood as a tool for fire investigations.
# PARAMETERS:
# average_heat_flux     q_bar   [W/m2]      time-averaged heat flux of the furnace internal from the test beginning
# timber_density        rho     [kg/m3]     density of timber
# exposure_time         t       [s]         time since the test beginning
# oxygen_factor         k_ox    [-]         oxygen factor; 1.0 = plentiful of o2, 0.8 = 8-10% o2, 0.55 = 4% o2 concentr.
def timber_charring_rate_babrauskas2005(average_heat_flux, timber_density, exposure_time, oxygen_content_volume=0.2):
    average_heat_flux /= 1000.  # convert from [W/m2] to [kW/m2]
    exposure_time /= 60.  # convert from [s] to [min]

    x = [1.00, 0.21, 0.12, 0.09, 0.04, 0.00, -1.e-6]
    y = [1.00, 1.00, 1.00, 0.80, 0.55, 0.55, 0.55]

    oxygen_factor = interp1d(x, y)(oxygen_content_volume)

    charring_rate = 0.
    if exposure_time != 0:
        charring_rate = 113. * oxygen_factor * np.sqrt(average_heat_flux) / timber_density / np.power(exposure_time, 0.3)
        # [mm/min]

    charring_rate /= 60000.  # convert from [mm/min] to [m/s]

    # print "---------------------------------"
    # print "exposure time:     " + str(np.round(exposure_time, 3))
    # print "average heat flux: " + str(np.round(average_heat_flux, 3))
    # print "oxygen content:    " + str(np.round(oxygen_content_volume, 3))
    # print "oxygen factor:     " + str(np.round(oxygen_factor, 3))
    # print "charring rate:     " + str(np.round(charring_rate, 6))

    return charring_rate


# DESCRIPTION:
# Calculates the current temperature for layers based on available information (i.e. k, rho, c, T and x). These
# available information is presumably from the previous time step.
# This function returns a ndarray which contains temperatures of different layers, i.e. [T_0, T_1, T_2, ... T_n].
# INPUT PARAMETERS:
# layers_conductivity   k       [W/m/K]     thermal conductivity
# layers_density        ρ       [kg/m3]     density of boundary material
# layers_specific_heat  c       [J/kg/K]    specific heat of boundary material
# layers_temperature    T       [K]         temperature of boundary material (previous time step)
# layers_thickness      dx      [m]         virtual layers' thickness of boundary layer
# time_step             dt      [s]         time step
# inner_conductivity    k_in    [W/m/K]     overall dQ/dt at j = -1 side (interior)
# inner_temperature     T_in    [K]         temperature at j = -1 side (interior)
# outer_conductivity    k_ou    [W/m/K]     overall dQ/dt at j = n+1 side (exterior)
# outer_temperature     T_ou    [K]         temperature of at j = n+1 side (exterior)
# NOTE:
# inert boundary - make inner_conductivity = 0.5 * x[0] / k[0] or outer_conductivity = 0.5 * x[n] / k[n]
def heat_transfer_general_1d_finite_difference(
        layers_conductivity,
        layers_density,
        layers_specific_heat,
        layers_temperature,
        layers_thickness,
        time_step,
        inner_conductivity,
        inner_temperature,
        outer_conductivity,
        outer_temperature
):
    # =================================================== PERMEABLE ====================================================
    # refactor parameters for readability
    k = -layers_conductivity
    rho = layers_density
    c = layers_specific_heat
    T = layers_temperature
    x = layers_thickness
    dt = time_step
    k_in = -inner_conductivity
    T_in = inner_temperature
    k_ou = -outer_conductivity
    T_ou = outer_temperature

    # instantiate returning parameter T_i, current temperature at all virtual layers
    dT_dt = np.zeros(shape=np.shape(T), dtype=float)

    # ====================================== FINITE DIFFERENCE METHOD CALCULATION ======================================
    for j in np.arange(0, len(layers_conductivity)):  # go through every virtual layers
        if j == 0:  # surface (inner) layer adjacent to x = 0
            dT1 = T[j] - T_in
            dT2 = T[j+1] - T[j]
            k_dx1 = 0.5 * 1/k[j] * x[j] + 1/k_in
            k_dx2 = 0.5 * 1/k[j+1] * x[j+1] + 0.5 * 1/k[j] * x[j]

        elif j == len(layers_conductivity) - 1:  # surface (outer) layer adjacent to x = max
            dT1 = T[j] - T[j-1]
            dT2 = T_ou - T[j]
            k_dx1 = 0.5 * 1/k[j] * x[j] + 0.5 * 1/k[j-1] * x[j-1]
            k_dx2 = 1/k_ou + 0.5 * 1/k[j] * x[j]

        else:  # intermediate layers
            dT1 = T[j] - T[j-1]
            dT2 = T[j+1] - T[j]
            k_dx1 = 0.5 * 1/k[j] * x[j] + 0.5 * 1/k[j-1] * x[j-1]
            k_dx2 = 0.5 * 1/k[j+1] * x[j+1] + 0.5 * 1/k[j] * x[j]

        # in case k_dx is zero (i.e. k[j] + k[j-1] = 0, sometime used for symmetry or inert boundary)
        k_dT_dx1 = dT1 / k_dx1 if k_dx1 != 0 else 0
        k_dT_dx2 = dT2 / k_dx2 if k_dx2 != 0 else 0
        dT_dt[j] = (k_dT_dx1 - k_dT_dx2) / c[j] / rho[j] / x[j]

    return dT_dt


# DESCRIPTION:
# This function calculates the amount of specified chemical (molecule_type_str) when burning a type of fuel
# (fuel_type_str) which associated with generating certain energy (energy_required_joule).
# INPUT PARAMETERS:
# energy_required_joule     Q   [J]     energy that is generated
# fuel_type_str             -   [-]     a string describing fuel type, it can be:
#                                       'propane', 'glucose' and 'cellulose'
# molecule_type_str         -   [-]     molecule type which involves in the combustion chemistry:
#                                       'fuel required', 'O2 required', 'CO2 produced' and 'H2O produced'
# return                    m   [kg]    mass of described molecule type
def combustion_chemistry(energy_required_joule, fuel_type_str, molecule_type_str):
    mass_fuel_required_gram = None
    mass_o2_required_gram = None

    # values required for generating 1000 J of energy
    dict_fuel_require = {  # heat of combustion (fuel), unit: [J/kg]
        "propane": 46.450e6,
        "glucose": 15.400e6,
        "cellulose": 16.090e6,
        "methane":50.e6,
    }
    dict_o2_require = {  # heat of combustion (o2), unit: [J/kg]
        "propane": 12.800e6,
        "glucose": 13.270e6,
        "cellulose": 13.590e6,
        "methane":12.54e6,
    }

    dict_co2_produce = {  # energy generated per CO2 kg, unit: [J/kg]
        "propane": 12.800e6 * ((32.*5.)/(44.01*3.)),  # O2 to CO2 ratio - 5:3
        "glucose": 13.270e6 * ((32.*1.)/(44.01*1.)),  # O2 to CO2* ratio - 1:1 (C6H12O6)
        "cellulose": 13.590e6 * ((32.*1.)/(44.01*1.)),  # O2 to CO2* ratio - 1:1 (C6H12O6)
        "methane":12.54e6 * ((32.*2.)/(44.*1.)),  # O2 : CO2 - 2 : 1
    }

    dict_h2o_produce = {  # energy generated per H2O kg, unit: [J/kg]
        "propane": 12.800e6 * ((32.00*5.)/(18.20*4.)),  # O2 to H2O ratio - 5:4
        "glucose": 13.270e6 * ((32.00*1.)/(18.02*1.)),  # O2 to C6H12O6 ratio - 1:1
        "cellulose": 13.590e6 * ((32.00*1.)/(18.02*1.)),  # 6*O2 to 6*H2O ratio - 1:1 (assumed C6H12O6)
        "methane": 12.54e6 * ((32.00*1.)/(18.02)),  # O2 : H2O - 1 : 1
    }
    # * indicates it is a chain of molecules

    dict_to_use = None
    if molecule_type_str == "fuel required":
        dict_to_use = dict_fuel_require
    elif molecule_type_str == "O2 required":
        dict_to_use = dict_o2_require
    elif molecule_type_str == "CO2 produced":
        dict_to_use = dict_co2_produce
    elif molecule_type_str == "H2O produced":
        dict_to_use = dict_h2o_produce
    else:
        print("Warning: Invalid molecule_str name.")

    return energy_required_joule / dict_to_use[fuel_type_str]  # [kg]


def output_numerical(data_dict, sample_index_interval = 60.):
    # data_dict["temperature_lining"] = data_dict["temperature_lining"][:,0]
    # data_dict["temperature_specimen"] = data_dict["temperature_specimen"][:,0]

    data_list = list()
    data_list2 = list()
    for key, val in sorted(data_dict.iteritems()):
        key_ = str(key)

        val_ = np.round(val, 6)

        if key == "temperature_lining" or key == "temperature_specimen":
            val_ = val_[-1, :]

        val_2 = val_
        val_2 = np.ma.masked_where(data_dict['time'] % sample_index_interval != 0, val_2)

        val_ = val_.tolist()
        val_2 = np.ma.compressed(val_2).tolist()

        val_.insert(0, key_)
        val_2.insert(0, key_)

        data_list.append(val_)
        data_list2.append(val_2)

    data_numerical = zip(*data_list)
    data_numerical2 = zip(*data_list2)

    return data_numerical, data_numerical2


def output_analysis(data_dict):
    t = data_dict['time']
    dt = t[1] - t[0]

    def a(label_str, arr, x, eng_format, unit_str):
        tot_ = round(np.trapz(arr,x) / (10.**eng_format), 0)
        ave_ = round(np.average(arr) / (10.**eng_format), 0)
        peak_ = round(np.max(np.abs(arr)) / (10. ** eng_format), 0)
        return label_str, tot_, ave_, peak_, unit_str

    hr_burner = data_dict['hr_burner_combustion']
    hr_specimen_combustion = data_dict['hr_specimen_combustion']
    hr_overall = hr_burner + hr_specimen_combustion
    hr_lining = data_dict['hr_lining']
    hr_ventilation = data_dict['hr_ventilation']
    hr_intern_gas = data_dict['hr_gas_enclosure']
    hr_specimen = data_dict['hr_specimen']
    hr_window = data_dict['hr_windows']

    hf_instantaneous = data_dict['hf_furnace']
    hf_averaged = data_dict['hf_furnace_average']

    mr_fuel = data_dict['mass_fuel_in']
    mr_specimen = data_dict['mass_specimen_in']
    mr_ventilation_in = data_dict['mass_ventilation_in']
    mr_ventilation_ou = data_dict['mass_ventilation_ou']

    temperature_gas = data_dict['temperature_gas']
    temperature_lining = data_dict['temperature_lining'][-1]
    temperature_specimen = data_dict['temperature_specimen'][-1]

    time_deviation = np.sum(data_dict['temperature_gas']!=data_dict['temperature_gas_target']) * dt
    time_ventilation_control = np.sum(np.gradient(data_dict['hr_specimen_combustion'])!=0.) * dt

    lr_specimen = data_dict['length_specimen_charring']
    content_vol_O2 = data_dict['content_vol_O2']

    data_analysis_dict = [
        ("VARIABLE", 'TOT.', 'AVE. [/s]', 'PEAK [/s]', 'UNIT'),
        a("Heat rate - overall", hr_overall, t, 3, '[kJ]'),
        a('Heat rate - burner', hr_burner, t, 3, '[kJ]'),
        a('Heat rate - specimen comb.', hr_specimen_combustion, t, 3, '[kJ]'),
        a('Heat rate - vent.', hr_ventilation, t, 3, '[kJ]'),
        a('Heat rate - lining', hr_lining, t, 3, '[kJ]'),
        a('Heat rate - specimen', hr_specimen, t, 3, '[kJ]'),
        a('Heat rate - intern. gas', hr_intern_gas, t, 3, '[kJ]'),
        a('Heat rate - window', hr_window, t, 3, '[kJ]'),
        ('', '', '', '', ''),
        a('Heat flux - instantaneous', hf_instantaneous, t, 3, '[kW/m2]'),
        a('Heat flux - average', hf_averaged, t, 3, '[kW/m2]'),
        ('', '', '', '', ''),
        a('Mass rate - fuel', mr_fuel, t, 0, '[kg]'),
        a('Mass rate - specimen', mr_specimen, t, 0, '[kg]'),
        a('Mass rate - ventilation in', mr_ventilation_in, t, 0, '[kg]'),
        a('Mass rate - ventilation ou', mr_ventilation_ou, t, 0, '[kg]'),
        ('', '', '', '', ''),
        ('Temperature - furnace', '-', round(np.average(temperature_gas-273.15)), round(np.max(temperature_gas-273.15)), '[deg C]'),
        ('Temperature - lining', '-', round(np.average(temperature_lining-273.15)), round(np.max(temperature_lining-273.15)), '[deg C]'),
        ('Temperature - specimen', '-',  round(np.average(temperature_specimen-273.15)), round(np.max(temperature_specimen-273.15)), '[deg C]'),
        ('', '', '', '', ''),
        ('Time - temperature deviation', round(time_deviation/60.,1), '-', '-', '[min]'),
        ('Time - ventilation controlled', round(time_ventilation_control/60.,1), '-', '-', '[min]'),
        ('', '', '', '', ''),
        a('Length - charring', lr_specimen, t, -3, '[mm]'),
        a('Content - O2', content_vol_O2*1000., t, 0, '[‰ vol.]')
    ]

    return data_analysis_dict


def print_debug_information(form_str, values_arr, soffit_str=""):
    return 0
    output_string = soffit_str + "\n"
    for each_entry in values_arr:
        output_string += form_str.format(*each_entry)
    print(output_string)
    return output_string
