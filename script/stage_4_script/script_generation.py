from code.stage_4_code.Method_RNN_Gene import Method_RNN_Generalization
from code.stage_4_code.Dataset_Loader_Generation import Dataset_Loader
from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_4_code.Setting_Generation import Setting_Up_Run
from code.stage_4_code.Evaluator import Evaluate
import torch
import numpy as np

if 1:
    np.random.seed(2)
    torch.manual_seed(2)

    data_obj = Dataset_Loader('Joke Dataset', '')
    data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_generation/'
    data_obj.dataset_source_file_name = 'data'
    data_obj.load()

    method_obj = Method_RNN_Generalization('RNN', '', data_obj.vocab_w_to_ind, data_obj.vocab_ind_to_w)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_4_result/generation'
    result_obj.result_destination_file_name = 'generation_result'

    setting_obj = Setting_Up_Run('generation', '')

    evaluate_obj = Evaluate('generation', '')

    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    setting_obj.load_run_save_evaluate()
    print('************ Finish ************')