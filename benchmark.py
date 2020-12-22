import keras
from fine_pruning_model import BaselineModel
from utils import *

def random_dropout_model():
    repaired_model_filename = "data/random_dropout_model_G1.h5"
    poisoned_model_filename = "models/sunglasses_bd_net.h5"

    bd_model = keras.models.load_model(poisoned_model_filename)
    repaired_model = keras.models.load_model(repaired_model_filename)
    model = BaselineModel(poisoned=bd_model, repaired=repaired_model, N=1283)

    e = Evaluator(model)
    e.evaluate()

def fine_pruning_model():
    repaired_model_filename = "data/random_dropout_model_G1.h5"
    poisoned_model_filename = "models/sunglasses_bd_net.h5"

    bd_model = keras.models.load_model(poisoned_model_filename)
    repaired_model = keras.models.load_model(repaired_model_filename)
    model = BaselineModel(poisoned=bd_model, repaired=repaired_model, N=1283)

    e = Evaluator(model)
    e.evaluate()

if __name__ == '__main__':
    result = []
    random_dropout_model()
    fine_pruning_model()