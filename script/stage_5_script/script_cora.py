from code.stage_5_code.Dataset_Loader import Dataset_Loader
from code.stage_5_code.Method_GCN_Cora import Method_GCN_Cora
from code.stage_5_code.Setting_Up import Setting_Up_Run
from code.stage_5_code.Evaluator import Evaluate
from code.stage_5_code.Result_Saver import Result_Saver
from code.stage_5_code.Result_Loader import Result_Loader
import torch
import numpy as np

#---- Classification script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('cora', '')
    data_obj.dataset_source_folder_path = '../../data/stage_5_data/'
    data_obj.dataset_source_file_name = 'cora'

    method_obj = Method_GCN_Cora('GCN', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_5_result/cora'
    result_obj.result_destination_file_name = 'cora_result'

    setting_obj = Setting_Up_Run('cora', '')

    evaluate_obj = Evaluate('cora', '')

    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    setting_obj.load_run_save_evaluate()
    print('************ Finish ************')