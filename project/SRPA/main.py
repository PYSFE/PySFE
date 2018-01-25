# -*- coding: utf-8 -*-
import os, sys, copy, warnings

try:
    import project.SRPA.func_time_equivalence as teq
except ModuleNotFoundError:
    import sys

    __path_cwf = os.path.realpath(__file__)
    __dir_project = os.path.dirname(os.path.dirname(__path_cwf))
    __dir_project = os.path.dirname(__dir_project)

    sys.path.insert(0, __dir_project)

    import project.SRPA.func_time_equivalence as teq

_strfmt_1_1 = "{:25}{}"

if __name__ == "__main__":
    # SET WORK DIRECTORY
    project_full_paths = []
    if len(sys.argv) > 1:  # Check for system argument

        project_full_paths = copy.copy(sys.argv)
        del project_full_paths[0]

    else:  # Ask user to input work directory if no system arguments provided

        number_of_paths = input("How many directories: ")
        number_of_paths = int(number_of_paths)

        for i in range(number_of_paths):
            project_full_path = input("Work directory: ")
            project_full_paths.append(project_full_path)

    # remove quote symbol from input directory
    project_full_paths = [i.replace('"', '') for i in project_full_paths]
    project_full_paths = [i.replace("'", '') for i in project_full_paths]

    # convert to absolute path
    project_full_paths = [os.path.abspath(i) for i in project_full_paths]

    # check if user provided directories exist and remove invalid directories
    project_full_paths_copy = copy.copy(project_full_paths)
    for i,v in enumerate(project_full_paths_copy):
        if not os.path.isdir(v):
            warnings.warn('Directory does not exist: {}'.format(v))
            del project_full_paths[i]

    # MAIN BODY

    for project_full_path in project_full_paths:
        list_files = teq.step0_parse_input_files(dir_work=project_full_path)
        print(_strfmt_1_1.format("ENTRY", "CONTENT"))
        print(_strfmt_1_1.format("Work directory:", project_full_path))
        ff = "{} - {}"
        for f in list_files:
            id_ = os.path.basename(f).split(".")[0]
            teq.step1_inputs_maker(f)
            # teq.step2_main_calc(os.path.join(project_full_path, ff.format(id_, "args_main.p")), 5)
            # teq.step3_results_numerical(os.path.join(project_full_path, ff.format(id_, "res_df.p")))
            teq.step4_results_visulisation(os.path.join(project_full_path, ff.format(id_, "res_df.p")))
            teq.step6_results_visualization_temperature(os.path.join(project_full_path, ff.format(id_, "res_df.p")))
            teq.step7_select_fires_teq(project_full_path, id_)
            print("")

        teq.step5_results_visulisation_all(project_full_path)
