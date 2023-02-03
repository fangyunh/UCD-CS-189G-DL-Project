'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        acc_score = accuracy_score(self.data['true_y'], self.data['pred_y'])
        pre_score = precision_score(self.data['true_y'], self.data['pred_y'], average='weighted', zero_division=1)
        rec_score = recall_score(self.data['true_y'], self.data['pred_y'], average='weighted')
        f_score = f1_score(self.data['true_y'], self.data['pred_y'], average='weighted')
        return {"Accuracy Score": acc_score,
                "Precision Score": pre_score,
                "Recall Score": rec_score,
                "F1 Score": f_score}
        