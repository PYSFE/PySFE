# -*- coding: utf-8 -*-
import os
import project.SRPA.func_time_equivalence as teq

strformat_1_1 = "{:25}{}"

if __name__ == "__main__":
    # SETTINGS
    project_full_path = r"test"

    # ROUTINES
    project_full_path = os.path.abspath(project_full_path)
    list_files = teq.step0_parse_input_files(dir_work=project_full_path)
    print(strformat_1_1.format("ENTRY", "CONTENT"))
    print(strformat_1_1.format("Work directory:", project_full_path))
    ff = "{} - {}"
    for f in list_files:
        id_ = os.path.basename(f).split(".")[0]
        teq.step1_inputs_maker(f)
        teq.step2_main_calc(os.path.join(project_full_path, ff.format(id_, "args_main.p")), 5)
        teq.step3_results_numerical(os.path.join(project_full_path, ff.format(id_, "res_df.p")))
        teq.step4_results_visulisation(os.path.join(project_full_path, ff.format(id_, "res_df.p")))
        teq.step6_results_visualization_temperature(os.path.join(project_full_path, ff.format(id_, "res_df.p")))
        teq.step7_select_fires_teq(project_full_path, id_)
        print("")

    teq.step5_results_visulisation_all(project_full_path)
