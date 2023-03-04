from code.base_class.setting import setting
import numpy as np


class Setting_Up_Run(setting):

    def load_run_save_evaluate(self):
        loaded_data, dictionary = self.dataset.load()

        X_train, X_test = np.array(loaded_data['X_train']), np.array(loaded_data['X_test'])
        y_train, y_test = np.array(loaded_data['y_train']), np.array(loaded_data['y_test'])

        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        self.method.vocabulary_size = len(dictionary) + 2
        self.method.dictionary = dictionary
        learned_result = self.method.run()

        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        print(self.evaluate.evaluate())