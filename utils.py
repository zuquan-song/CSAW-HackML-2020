import numpy as np
import h5py



def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255


clean_test_filename = 'data/clean_test_data.h5'
poisoned_data_filename = 'data/sunglasses_poisoned_data.h5'


"""
    Model Requirement:
        model should have a method named "predict", which returns the predict class (NOT probabilities of all classes)
    Usage: 
        e = Evaluator(model)
        e.evaluate()
"""
class Evaluator:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

    def evaluate(self):
        # evaluate model
        clean_x, clean_y = data_loader(clean_test_filename)
        clean_x = data_preprocess(clean_x)

        poisoned_x, poisoned_y = data_loader(poisoned_data_filename)
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