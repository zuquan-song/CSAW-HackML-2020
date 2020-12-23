import keras
from fine_pruning_model import *
from auto_encoder_utils import AutoEncoderRepairedModel, AutoEncoder

"""
python autoencoder_based_model_eval.py [test_data_filename] [poisoned_model_filename] [repaired_model_filename]
eg: python autoencoder_based_model_eval.py data/clean_test_data.h5 fixed_models/fine_pruning_model_for_anonymous_1_bd_net.h5
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
    repaired_model = keras.models.load_model(repaired_model_filename)
    model = AutoEncoderRepairedModel(AutoEncoder(), repaired_model)

    x_test, y_test = data_loader(clean_data_filename)
    x_test = data_preprocess(x_test)
    clean_label_p = model.predict(x_test)
    class_accu = np.mean(np.equal(clean_label_p, y_test))*100
    print('Classification accuracy:', class_accu)


if __name__ == '__main__':
    clean_data_filename = str(sys.argv[1])
    repaired_model_filename = str(sys.argv[2])
    main()
