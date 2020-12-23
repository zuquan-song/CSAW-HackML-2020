import keras
from fine_pruning_model import *

class RuleBasedModel:
    def __init__(self, *args, **kwargs):
        self.origin_model = kwargs['poisoned']
        self.repaired_model = kwargs['repaired']
        self.N = kwargs['N']

    def predict(self, data):
        origin_result = np.argmax(self.origin_model.predict(data), axis=1)
        repaired_result = np.argmax(self.repaired_model.predict(data), axis=1)
        return np.array([a if a == b else self.N for a, b in zip(origin_result, repaired_result)])

"""
python rule_based_model_eval.py [test_data_filename] [poisoned_model_filename] [repaired_model_filename]
eg: python rule_based_model_eval.py data/clean_test_data.h5 models/sunglass_bd_net.h5 data/repaired_random_pruning_model_G1.h5
"""


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255


def main():
    bd_model = keras.models.load_model(poisoned_model_filename)
    repaired_model = keras.models.load_model(repaired_model_filename)
    model = RuleBasedModel(poisoned=bd_model, repaired=repaired_model, N=1283)

    x_test, y_test = data_loader(clean_data_filename)
    x_test = data_preprocess(x_test)
    clean_label_p = model.predict(x_test)
    class_accu = np.mean(np.equal(clean_label_p, y_test))*100
    print('Classification accuracy:', class_accu)


if __name__ == '__main__':
    clean_data_filename = str(sys.argv[1])
    poisoned_model_filename = str(sys.argv[2])
    repaired_model_filename = str(sys.argv[3])
    main()
