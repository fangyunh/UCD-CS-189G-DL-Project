
from code.base_class.setting import setting

class Setting_Up_Run(setting):

    def load_run_save_evaluate(self):
        context, targets, vocab_w_i, vocab_i_w, jokes, jokes_len = self.dataset.load()
        self.method.data = {'X': context, 'y': targets, 'w_i' : vocab_w_i, 'i_w' : vocab_i_w,
                            'jokes' : jokes, "jokes_len" : jokes_len}
        learned_result = self.method.run()

        self.result.data = learned_result
        self.result.save()