'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate(evaluate):
    data = None

    def evaluate(self):
        print('evaluating performance...')
        acc = accuracy_score(self.data['true_y'], self.data['pred_y'])
        precision = precision_score(self.data['true_y'], self.data['pred_y'], average='weighted', zero_division=1)
        recall = recall_score(self.data['true_y'], self.data['pred_y'], average='weighted', zero_division=1)
        f1 = f1_score(self.data['true_y'], self.data['pred_y'], average='weighted', zero_division=1)

        results = {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

        return results

