# -*- coding: utf-8 -*-
import os
from project.SPRS.func_time_equivalence import step0_parse_input_files, step1_inputs_maker, step2_main_calc, step3_results_numerical, step4_results_visulisation, step5_results_visulisation_all, step6_fire_curves_pick


if __name__ == "__main__":
    # SETTINGS
    project_full_path = r"C:\Users\ian\Desktop\trail 30mins"

    # ROUTINES
    project_full_path = os.path.abspath(project_full_path)
    list_files = step0_parse_input_files(dir_work=project_full_path)
    ff = "{} - {}"
    for f in list_files:
        id_ = f.split(".")[0]
        step1_inputs_maker(f)
        step2_main_calc(os.path.join(project_full_path, ff.format(id_, "args_main.p")), 5)
        step3_results_numerical(os.path.join(project_full_path, ff.format(id_, "res_df.p")))
        # step4_results_visulisation(os.path.join(project_full_path, ff.format(id_, "res_df.p")))
        print("")

    step5_results_visulisation_all(project_full_path)
