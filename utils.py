import numpy as np
import h5py
import keras

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255


def pruning(p, weights):
    thresh = np.percentile(weights[-1], p)
    super_threshold_indices = weights[-1] < thresh
    weights[-1][super_threshold_indices] = 0
    return weights

"""
    Model Requirement:
        model should have a method named "predict", which returns the predict class (NOT probabilities of all classes)
    Usage: 
        e = Evaluator(model)
        e.evaluate()
"""
class EvaluateResult(object):

    def __init__(self, **kwargs):
        self.kv = {}
        for k, v in kwargs.items():
            self.kv[k] = v

class Evaluator:

    def __init__(self, model_path=None, model = None, model_name='default', clean_file=None, poisoned_file=None):
        if model:
            self.model = model
        else:
            self.model = keras.models.load_model(model_path)
        self.model_name = model_name
        self.clean_file = clean_file
        self.poisoned_file = poisoned_file

    def evaluate(self):
        # evaluate model
        print("evaluate model: {}".format(self.model_name))
        clean_x, clean_y = data_loader(self.clean_file)
        clean_x = data_preprocess(clean_x)

        poisoned_x, poisoned_y = data_loader(self.poisoned_file)
        poisoned_x = data_preprocess(poisoned_x)

        # test model based on clean data
        y_result = self.model.predict(clean_x)
        accuracy = sum(y_result == clean_y) * 1.0 / len(clean_y)
        print("clean data accuracy: {}".format(accuracy))

        # test model based on poisoned data
        y_bar = self.model.predict(poisoned_x)
        attack_success_rate = sum(y_bar == poisoned_y) * 1.0 / len(poisoned_y)
        print("attack_success_rate: {}".format(attack_success_rate))
        trigger_detection_rate = sum(y_bar == 1283) * 1.0 / len(poisoned_y)
        print("trigger_detection_rate: {}".format(trigger_detection_rate))

        return EvaluateResult(
            model_name=self.model_name,
            clean_acc=accuracy,
            attack_succ_rate=attack_success_rate,
            trigger_detect_rate=trigger_detection_rate
        )