import os
import tensorflow as tf
import numpy as np
from architecture import *
import h5py
import tensorflow_model_optimization as tfmot
import tempfile
from keras.models import load_model


rdir = "."
poisoned_data_filename = 'data/sunglasses_poisoned_data.h5'
clean_validation_filename = 'data/clean_validation_data.h5'
clean_test_filename = 'data/clean_test_data.h5'


poisoned_model_filename = 'models/sunglasses_bd_net.h5'
poisoned_model_weights = 'models/sunglasses_bd_weights.h5'

repaired_model_filename = "repaired_model.h5"
repaired_model_weights = "repaired_model_weight.h5"

class BaselineModel:

    def __init__(self, *args, **kwargs):
        self.origin_model = kwargs['poisoned']
        self.repaired_model = kwargs['repaired']
        self.N = kwargs['N'] + 1

    def predict(self, data):
        origin_result = np.argmax(self.origin_model.predict(data), axis=1)
        repaired_result = np.argmax(self.repaired_model.predict(data), axis=1)
        return np.array([a if a == b else self.N for a, b in zip(origin_result, repaired_result)])


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255

if __name__ == '__main__':
    # bd_model = Net()
    bd_model = load_model(poisoned_model_filename)
    bd_model.load_weights(os.path.join(rdir, poisoned_model_weights))

    # original attack success
    poisoned_x, poisoned_y = data_loader(poisoned_data_filename)
    poisoned_x = data_preprocess(poisoned_x)
    clean_label_p = np.argmax(bd_model.predict(poisoned_x), axis=1)
    class_accu = np.mean(np.equal(clean_label_p, poisoned_y))*100
    print('original attack success rate:', class_accu)

    num_images = 12830
    batch_size = 32
    epochs = 2
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                 final_sparsity=0.80,
                                                                 begin_step=0,
                                                                 end_step=end_step)
    }

    clean_x, clean_y = data_loader(clean_validation_filename)
    clean_x = data_preprocess(clean_x)

    # create repaired model
    def apply_pruning_to_dense(layer):
        if layer.name in ['fc_2']:
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
        return layer

    tmp_model = load_model(poisoned_model_filename)
    tmp_model.load_weights(os.path.join(rdir, poisoned_model_weights))
    repaired_model = tf.keras.models.clone_model(
        tmp_model,
        clone_function=apply_pruning_to_dense,
    )

    repaired_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                              loss=tf.keras.losses.sparse_categorical_crossentropy,
                              metrics=['accuracy'])
    callback = [
      tfmot.sparsity.keras.UpdatePruningStep(),
      tfmot.sparsity.keras.PruningSummaries(log_dir=tempfile.mkdtemp()),
    ]
    repaired_model.fit(
        clean_x,
        clean_y,
        epochs=5,
        callbacks=callback,
    )

    # create baseline model based on bd_model + repaired_model
    model = BaselineModel(poisoned=bd_model, repaired=repaired_model, N=1283)

    # test model based on clean data
    y_result = model.predict(clean_x)
    accuracy = sum(y_result == clean_y) * 1.0 / len(clean_y)
    print("accuracy: {}".format(accuracy))

    # test model based on poisoned data
    poisoned_x, poisoned_y = data_loader(poisoned_data_filename)
    poisoned_x = data_preprocess(poisoned_x)
    y_result = model.predict(poisoned_x)
    attack_success_rate = sum(y_result == poisoned_y) * 1.0 / len(poisoned_y)
    print("attack_success_rate: {}".format(attack_success_rate))
    trigger_detection_rate = sum(y_result == 1284) * 1.0 / len(poisoned_y)
    print("trigger_detection_rate: {}".format(trigger_detection_rate))
