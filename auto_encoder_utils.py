import keras
from utils import data_loader, data_preprocess
import numpy as np

auto_encoder_model_filename = "fixed_models/autoencoder.h5"


class AutoEncoder:
    def __init__(self):
        self.auto_encoder = keras.models.load_model(auto_encoder_model_filename)
        self.auto_encoder.compile(optimizer='adam', loss='mean_squared_error')
        self.threshold = 0.09179428238229165

    def reset_threshold(self, retrained_data_filename):
        retrained_x, retrained_y = data_loader(retrained_data_filename)
        retrained_x = data_preprocess(retrained_x)
        reconstructions = self.auto_encoder.predict(retrained_x)
        train_loss = keras.losses.mae(reconstructions, retrained_x)
        self.threshold = np.mean(train_loss) + np.std(train_loss) + 0.03

    def encoder_predict(self, input_x):
        '''
        Generate mask for detecting the abnomal input. Once the constrcture loss over the threhold,
        we treat it as abnormal input

        Args:
            autoencoder(Model): autoencoder model
            input_x (array size with [None,55,47,3]): Input data
            threshold (float): used to detect abnomal input

        Retrun:
            res: A mask like [0,1,0,1....] in numpy.array type. 0 is abnormal point, and 1 is validated point

        '''
        reconstruction = self.auto_encoder.predict(input_x)
        detect_loss = keras.losses.mae(reconstruction, input_x)
        res = np.fromiter(map(lambda x: 1 if (np.mean(x) + np.std(x)) <= self.threshold else 0, detect_loss), dtype=int)
        return res


class AutoEncoderRepairedModel:

    def __init__(self, autoencoder, repaired_model):
        self.AE = autoencoder
        self.repaired_model = repaired_model

    def predict(self, x):
        mask = self.AE.encoder_predict(x)
        y_hat = np.argmax(self.repaired_model.predict(x), axis=1)
        y_hat[mask == 0] = 1283
        return y_hat


'''
def final_predict(autoencoder, repaired_model, x):
    mask = autoencoder.encoder_predict(x)
    y_hat = np.argmax(repaired_model.predict(x), axis=1)
    y_hat[mask == 0] = 1283
    return y_hat


if __name__ == '__main__':

    AE = AutoEncoder()
    repaired_model = keras.models.load_model("fixed_models/pruning.h5")
    retrained_x, retrained_y = data_loader(retrained_data_filename)
    retrained_x = data_preprocess(retrained_x)
    retrained_y_hat =   final_predict(AE,repaired_model,retrained_x)
    print(np.mean(np.equal(retrained_y_hat, retrained_y)) * 100)

'''
