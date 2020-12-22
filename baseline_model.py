import os
import tensorflow as tf
import sys
import tensorflow_model_optimization as tfmot
import tempfile
from keras.models import load_model
from utils import *
import keras as K


retrained_data_filename = 'data/clean_validation_data.h5'
repaired_model_filename = "models/repaired_model_baseline.h5"


class BaselineModel:
    def __init__(self, *args, **kwargs):
        self.origin_model = kwargs['poisoned']
        self.repaired_model = kwargs['repaired']
        self.N = kwargs['N'] + 1

    def predict(self, data):
        origin_result = np.argmax(self.origin_model.predict(data), axis=1)
        repaired_result = np.argmax(self.repaired_model.predict(data), axis=1)
        return np.array([a if a == b else self.N for a, b in zip(origin_result, repaired_result)])


if __name__ == '__main__':
    clean_data_filename = str(sys.argv[1])
    model_filename = str(sys.argv[2])

    bd_model = load_model(model_filename)

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

    retrained_x, retrained_y = data_loader(retrained_data_filename)
    retrained_x = data_preprocess(retrained_x)

    # create repaired model
    def apply_pruning_to_dense(layer):
        if layer.name in ['fc_2']:
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
        return layer

    if not os.path.exists(repaired_model_filename):
        tmp_model = load_model(poisoned_model_filename)
        def apply_pruning_to_dense(layer):
            if layer.name in ['fc_2']:
                return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
            return layer
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
            retrained_x,
            retrained_y,
            epochs=10,
            callbacks=callback,
        )
        repaired_model = tfmot.sparsity.keras.strip_pruning(repaired_model)

        tf.keras.models.save_model(repaired_model, repaired_model_filename, include_optimizer=False)
        # repaired_model.save(repaired_model_filename)
    else:
        repaired_model = K.models.load_model(repaired_model_filename)
        repaired_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                         loss=tf.keras.losses.sparse_categorical_crossentropy,
                         metrics=['accuracy'])

    # create baseline model based on bd_model + repaired_model
    model = BaselineModel(poisoned=bd_model, repaired=repaired_model, N=1283)

    test_x, test_y = data_loader(clean_data_filename)
    test_x = data_preprocess(test_x)
    result_x = model.predict(test_x)

    class_accu = np.mean(np.equal(result_x, test_y))*100
    print('Classification accuracy:', class_accu)

