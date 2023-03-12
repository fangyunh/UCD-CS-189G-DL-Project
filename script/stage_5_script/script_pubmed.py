from code.stage_5_code.Dataset_Loader import Dataset_Loader
from code.stage_5_code.Method_GCN_Pubmed import Method_GCN_Pubmed
from code.stage_5_code.Setting_Up import Setting_Up_Run
from code.stage_5_code.Evaluator import Evaluate
from code.stage_5_code.Result_Saver import Result_Saver
import torch
import numpy as np

#---- Classification script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('pubmed', '')
    data_obj.dataset_source_folder_path = '../../data/stage_5_data/'
    data_obj.dataset_source_file_name = 'pubmed'

    method_obj = Method_GCN_Pubmed('GCN', '')

    result_obj = Result_Saver('Pubmed saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_5_result/pubmed'
    result_obj.result_destination_file_name = 'Pubmed_result'

    setting_obj = Setting_Up_Run('Pubmed', '')

    evaluate_obj = Evaluate('Pubmed', '')

    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    setting_obj.load_run_save_evaluate()
    print('************ Finish ************')