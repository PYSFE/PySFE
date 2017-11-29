# -*- coding: utf-8 -*-
import time
from project.zone_model_1.CALCS.FurnaceModel import *
from project.zone_model_1.CTS.CTS import write_list_to_csv as write_data
from project.zone_model_1.CTS.CTS import PlotScatter2D
from project.zone_model_1.CTS.CTS import list_all_files_with_suffix
import re
import os
import multiprocessing

version = "1.02"
time_epoch = 1483228800


def input_file_selection(folder_dir):
    # list all available input files and let user make the decision
    input_files_list = list_all_files_with_suffix(folder_dir, ".txt")

    disp_string = "{:12}{}".format("FILE NO.", "DIRECTORY")
    for i, v in enumerate(input_files_list):
        disp_string += "\n{:12}{}".format(str(i),v)
    print(disp_string)

    # assign selected input file
    if len(input_files_list) > 1:
        selected_file_no = int(raw_input("Please select input file (-1 to select all): "))
    else:
        print("Please select input file (-1 to select all): 0")
        selected_file_no = 0
    input_file_dir_list = []
    if selected_file_no > -1:
        input_file_dir_list.append(input_files_list[selected_file_no])
    else:
        input_file_dir_list = input_files_list

    return input_file_dir_list


def parse_data_from_file(input_file_dir):
    with open(input_file_dir) as input_file:
        input_text = input_file.read()

    # remove spaces
    input_text = input_text.replace(" ", "")

    # break raw input text to a list
    input_text = re.split(';|\r|\n', input_text)

    # delete comments (anything followed by #)
    input_text = [v.split("#")[0] for v in input_text]

    # delete empty entries
    input_text = [v for v in input_text if v]

    # for each entry in the list (input_text), break up i.e. ["variable_name=1+1"] to [["variable_name"], ["1+1"]]
    input_text = [v.split("=") for v in input_text]

    # transform the list (input_text) to a dictionary. i.e. [["variable_name"], ["1+1"]] to {"variable_name": 2}

    input_text_dict = {v[0]: eval(v[1].replace('_', ' ')) for v in input_text}

    return input_text_dict


def run_main_function(input_text_dict):
    results_dict = main_function(
        time_step=input_text_dict["time_step"],
        time_cap=input_text_dict["time_cap"],
        ambient_air_density=input_text_dict["ambient_air_density"],
        ambient_air_pressure=input_text_dict["ambient_air_pressure"],
        ambient_temperature=input_text_dict["ambient_temperature"],
        gas_emissivity=input_text_dict["gas_emissivity"],
        ventilation_modifier_fixed=input_text_dict["ventilation_modifier_fixed"],
        enclosure_volume=input_text_dict["enclosure_volume"],
        enclosure_gas_pressure=input_text_dict["enclosure_gas_pressure"],
        fuel_material=input_text_dict["fuel_material"],
        specimen_surface_emissivity=input_text_dict["specimen_surface_emissivity"],
        specimen_exposed_area=input_text_dict["specimen_exposed_area"],
        specimen_thickness=input_text_dict["specimen_thickness"],
        specimen_heat_of_combustion=input_text_dict["specimen_heat_of_combustion"],
        specimen_density=input_text_dict["specimen_density"],
        specimen_moisture_content=input_text_dict["specimen_moisture_content"],
        lining_surface_emissivity=input_text_dict["lining_surface_emissivity"],
        lining_surface_area=input_text_dict["lining_surface_area"],
        lining_thickness=input_text_dict["lining_thickness"],
        lining_conductivity=input_text_dict['lining_conductivity'],
        lining_density=input_text_dict["lining_density"],
        lining_specific_heat=input_text_dict["lining_specific_heat"],
        window_area=input_text_dict["window_area"]
    )
    return results_dict


def create_output_folder(output_folder_dir):
    if not os.path.exists(output_folder_dir):
        os.makedirs(output_folder_dir)


def save_numerical_data(results_dict, save_folder_dir,compress_factor, file_name="~data.csv"):
    output_full, output_compressed = output_numerical(results_dict, compress_factor)
    write_data(
        "{}/{}".format(save_folder_dir, file_name),
        output_compressed
    )


def save_analysed_data(results_dict, save_folder_dir, file_name="~analysis.csv"):
    write_data(
        '{}/{}'.format(save_folder_dir, file_name),
        output_analysis(results_dict)
    )


def generate_plot_data(results_dict, mode_full_mid_min):
    plot_data = []
    x = results_dict["time"] / 60.  # common x-values (time axis)
    # temperature (lining) ---------------------------------------------------------------------------------------------
    if mode_full_mid_min == "mid" or mode_full_mid_min == "full":
        xyl1 = [
        [x, results_dict["temperature_gas"]-273.15, "Gas temperature"]
        ]
        for i, v in enumerate(results_dict["temperature_lining"]):
            xyl1.append([x, v-273.15, "Lining temp. " + str(i)])
        axis_label_y1 = "Temperature [$\degree C$]"
        file_name = "temperature (lining)"

        plot_data.append({"xyl1": xyl1, "axis_label_y1": axis_label_y1, "file_name": file_name})

    # temperature (specimen) -------------------------------------------------------------------------------------------
    if mode_full_mid_min == "mid" or mode_full_mid_min == "full":
        xyl1 = [
        [x, results_dict["temperature_gas"]-273.15, "Gas temperature"]
        ]
        for i, v in enumerate(results_dict["temperature_specimen"]):
            xyl1.append([x, v-273.15, "Specimen temp. " + str(i)])
        axis_label_y1 = "Temperature [$\degree C$]"
        file_name = "temperature (specimen)"

        plot_data.append({"xyl1": xyl1, "axis_label_y1": axis_label_y1, "file_name": file_name})

    # heat -------------------------------------------------------------------------------------------------------------
    xyl1 = [
        [x, results_dict["hr_specimen_combustion"] / 1.e6, "Heat release rate from timber combustion"],
        [x, results_dict["hr_burner_combustion"] / 1.e6, "Heat release rate from burner"],
        [x, results_dict["hr_lining"] / 1.e6, "Heat loss rate to lining"],
        [x, results_dict["hr_specimen"] / 1.e6, "Heat loss rate to specimen"],
        [x, results_dict["hr_windows"] / 1.e6, "Heat loss rate to windows"],
        [x, results_dict["hr_ventilation"] / 1.e6, "Heat loss rate to ventilation"],
        [x, results_dict["hr_gas_enclosure"] / 1.e6, "Heat loss rate to internal gas"],
    ]
    axis_label_y1 = "Heat rate [$MW$]"
    file_name = "energy rate"

    plot_data.append({"xyl1": xyl1, "axis_label_y1": axis_label_y1, "file_name": file_name})

    # mass -------------------------------------------------------------------------------------------------------------
    if mode_full_mid_min == "full":
        xyl1 = [
            [x, results_dict["mass_fuel_in"], "Mass burning rate of fuel"],
            [x, results_dict["mass_specimen_in"], "Mass burning rate of specimen"],
            [x, results_dict["mass_ventilation_in"], "Mass flow rate of ventilation (inlet)"],
            [x, results_dict["mass_ventilation_ou"], "Mass flow rate of ventilation (outlet)"]
        ]
        axis_label_y1 = "Mass rate [$kg/s$]"
        file_name = "mass rate"

        plot_data.append({"xyl1": xyl1, "axis_label_y1": axis_label_y1, "file_name": file_name})

    # c_p --------------------------------------------------------------------------------------------------------------
    if mode_full_mid_min == "full":
        xyl1 = [
            [x, results_dict["c_O2"], "Specific heat of $O_2$"],
            [x, results_dict["c_CO2"], "Specific heat of $CO_2$"],
            [x, results_dict["c_N2"], "Specific heat of $N_2$"],
            [x, results_dict["c_Ar"], "Specific heat of $Ar$"],
            [x, results_dict["c_H2O"], "Specific heat of $H_2O$"]
        ]
        axis_label_y1 = "Specific heat [$J/kg/K$]"
        file_name = "specific heat"

        plot_data.append({"xyl1": xyl1, "axis_label_y1": axis_label_y1, "file_name": file_name})

    # charring rate ----------------------------------------------------------------------------------------------------
    if mode_full_mid_min == "full":
        xyl1 = [
            [x, results_dict["length_specimen_charring"] * 60000., "Charring rate"]  # convert from [m/s] to [mm/min]
        ]
        axis_label_y1 = "Charring rate [$mm/min$]"
        file_name = "charring rate"

        plot_data.append({"xyl1": xyl1, "axis_label_y1": axis_label_y1, "file_name": file_name})

    # heat flux --------------------------------------------------------------------------------------------------------
    if mode_full_mid_min == "mid" or mode_full_mid_min == "full":
        xyl1 = [
            [x, results_dict["temperature_gas_target"]-273.15, "ISO 834 temperature"],
            [x, results_dict["temperature_gas"]-273.15, "Furnace temperature"]
        ]
        xyl2 = [
            [x, results_dict["hf_furnace"] / 1.e6, "Furnace heat flux (instantaneous)"],
            [x, results_dict["hf_furnace_average"] / 1.e6, "Furnace heat flux (average)"]
        ]
        axis_label_y1 = "Temperature [$\degree C$]"
        axis_label_y2 = "Heat flux [$MW/m^2$]"
        file_name = "temperature and heat flux"

        plot_data.append({"xyl1": xyl1, "axis_label_y1": axis_label_y1, "xyl2": xyl2, "axis_label_y2": axis_label_y2,
                      "file_name": file_name})

    # species content (burner combustion only) -------------------------------------------------------------------------
    if mode_full_mid_min == "mid" or mode_full_mid_min == "full":
        xyl1 = [
            [x, results_dict["content_vol_N2"] * 100., "N2 % by volume"],
            [x, results_dict["content_vol_O2"] * 100., "O2 % by volume"],
            [x, results_dict["content_vol_Ar"] * 100., "Ar % by volume"],
            [x, results_dict["content_vol_CO2"] * 100., "CO2 % by volume"],
            [x, results_dict["content_vol_H2O"] * 100., "H2O % by volume"],
            [x, results_dict["content_vol_CH4"] * 100., "CH4 % by volume"]
        ]
        axis_label_y1 = "Content [$\%\\ m^{3}$]"
        file_name = "content - species"

        plot_data.append({"xyl1": xyl1, "axis_label_y1": axis_label_y1, "file_name": file_name})

    # fuel combustion chemistry ----------------------------------------------------------------------------------------
    if mode_full_mid_min == "full":
        xyl1 = [
            [x, results_dict["mass_fuel_in"], "Fuel burning rate"],
            [x, results_dict["mass_o2_burner_required"], "O2 required by fuel combustion"],
            [x, results_dict["mass_co2_burner_produced"], "CO2 produced by fuel combustion"],
            [x, results_dict["mass_h2o_burner_produced"], "H2O produced by fuel combustion"],
        ]
        axis_label_y1 = "Mass flow rate [$m^3/s$]"
        file_name = "combustion fuel"

        plot_data.append({"xyl1": xyl1, "axis_label_y1": axis_label_y1, "file_name": file_name})

    # timber combustion chemistry --------------------------------------------------------------------------------------
    if mode_full_mid_min == "full":
        xyl1 = [
            [x, results_dict["mass_specimen_in"], "Specimen burning rate"],
            [x, results_dict["mass_o2_specimen_required"], "O2 required by specimen combustion"],
            [x, results_dict["mass_co2_specimen_produced"], "CO2 produced by specimen combustion"],
            [x, results_dict["mass_h2o_specimen_produced"], "H2O produced by specimen combustion"],
        ]
        axis_label_y1 = "Mass flow rate [$kg/s$]"
        file_name = "combustion specimen"

        plot_data.append({"xyl1": xyl1, "axis_label_y1": axis_label_y1, "file_name": file_name})

    # Finish
    return plot_data


def plot_graphs(plot_data, folder_dir, file_suffix, default_format, suppress_viewing=False):
    figs = []
    for i, v in enumerate(plot_data):
        fig = PlotScatter2D()
        xyl1 = v["xyl1"] if "xyl1" in v else None
        xyl2 = v["xyl2"] if "xyl2" in v else None
        fig.plot(xyl1, xyl2)
        axis_label_y1 = v["axis_label_y1"] if "axis_label_y1" in v else None
        axis_label_y2 = v["axis_label_y2"] if "axis_label_y2" in v else None
        fig.format(
            axis_label_y1=axis_label_y1,
            axis_label_y2=axis_label_y2,
            **default_format
        )
        fig.format_figure(**default_format)
        fig.update_legend(**default_format)

        if not suppress_viewing:
            fig.figure.show()

        figs.append(fig)

    if not suppress_viewing:
        raw_input("Enter any key to close and save plots and finish the model.")

    for i, v in enumerate(figs):
        v.save_figure(
            figure_name="{}/{}".format(folder_dir, plot_data[i]["file_name"] + "{}".format(file_suffix))
        )
        v.self_delete()


def wrapped_all_calcs(i, input_file_dir, table_string, compress_factor, mode_ful_mid_min="full"):
    file_dir_ = input_file_dir.split("/")
    file_name = file_dir_[len(file_dir_)-1]
    del file_dir_[len(file_dir_)-1]
    file_folder = "/".join(file_dir_)
    trial_str = file_name.split(" ")
    trial_str = " " + trial_str[len(trial_str)-1].replace(".txt","")

    print(table_string.format(str(i), "-", "Running"))
    time_start = time.time()

    input_text_dict = parse_data_from_file(input_file_dir=input_file_dir)

    results, status = run_main_function(input_text_dict=input_text_dict)

    create_output_folder(output_folder_dir=file_folder)

    save_numerical_data(
        results_dict=results,
        save_folder_dir=file_folder,
        compress_factor=compress_factor,
        file_name="~data{}.csv".format(trial_str)
    )

    save_analysed_data(results_dict=results, save_folder_dir=file_folder, file_name="~analysis{}.csv".format(trial_str))

    plot_data = generate_plot_data(results_dict=results, mode_full_mid_min=mode_ful_mid_min)

    default_format = {
        "figure_size_width": 4.5,
        "figure_size_height": 4.5*0.75,
        "figure_size_scale": 1.,
        "mark_every": len(results["time"]) / 10,
        "marker_size": 2,
        "axis_label_x": "Time [$min$]",
        "axis_lim_x": [0, max(results["time"]/60.)],
        "axis_label_font_size": 11.,
        "axis_tick_font_size": 9.,
        "axis_linewidth": 1.,
        "axis_tick_width": 0.6,
        "axis_tick_length": 2.5,
        "legend_is_fancybox": True,
        "legend_alpha": 0.8,
        "legend_font_size": 6.,
        "legend_line_width": 0.25,
        "legend_is_shown": False,
        "line_width": .75,
        "marker_edge_with": 0.5,
    }

    plot_graphs(
        plot_data=plot_data,
        folder_dir=file_folder,
        file_suffix=trial_str,
        default_format=default_format,
        suppress_viewing=True
    )

    print(table_string.format(str(i), str(int(time.time()-time_start))+"s", status))


def main_hub():
    mode_ful_mid_min = "full"
    number_of_process = 4
    work_folder_dir = "OUTPUTS/demo/"
    compress_factor = 0.5

    print(chr(27) + "[2J")  # clear terminal window
    print("*"*25+"\nFurnaceModel version {:4}\n".format(version)+"*"*25+"\n")

    input_file_dir_list = input_file_selection(folder_dir=work_folder_dir)

    print("")
    table_string = "{:12}{:14}{:}"
    print(table_string.format("FILE NO.", "TIME TAKEN", "STATUS"))

    jobs = []
    for i,v in enumerate(input_file_dir_list):
        p = multiprocessing.Process(target=wrapped_all_calcs, args=(i, v, table_string, compress_factor, mode_ful_mid_min))
        jobs.append(p)
        p.start()
        time.sleep(0.5)

    for job in jobs:
        job.join()

    print("\nAll jobs are done - FurnaceModel version {:4}\n".format(version))

if __name__ == "__main__":
    main_hub()
