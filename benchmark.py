
import keras
from baseline_model import BaselineModel
from utils import *
def baseline_model():
    repaired_model_filename = "data/repaired_model_baseline.h5"
    poisoned_model_filename = "models/sunglasses_bd_net.h5"

    bd_model = keras.models.load_model(poisoned_model_filename)
    repaired_model = keras.models.load_model(repaired_model_filename)
    model = BaselineModel(poisoned=bd_model, repaired=repaired_model, N=1283)

    e = Evaluator(model)
    e.evaluate()


if __name__ == '__main__':
    baseline_model()
