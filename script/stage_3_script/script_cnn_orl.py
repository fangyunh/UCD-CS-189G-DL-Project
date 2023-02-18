from code.stage_3_code.Dataset_Loader_ORL import Dataset_Loader
from code.stage_3_code.Result_Saver import Result_Saver
from code.stage_3_code.Setting_Up import Setting_Up
from code.stage_3_code.Evaluator import Evaluate
from code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
import torch
import numpy as np

if 1:
    # ####
    np.random.seed(2)
    torch.manual_seed(2)

    data_obj = Dataset_Loader('orl', '')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    data_obj.dataset_source_file_name = 'ORL'

    method_obj = Method_CNN_ORL('ORL', '')

    result_obj = Result_Saver('save', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/ORL'
    result_obj.result_destination_file_name = 'result'

    setting_obj = Setting_Up('ORL_Setting', '')

    evaluate_obj = Evaluate('eva', '')

    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    setting_obj.load_run_save_evaluate()
    print('************ Finish ************')
    # ------------------------------------------------------
