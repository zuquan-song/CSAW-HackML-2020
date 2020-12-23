import pandas as pd
from fine_pruning_model import FinePruningModel
from random_pruning_model import RandomPruningModel
from auto_encoder_utils import *
from utils import *
import matplotlib.pyplot as plt

repaired_model_filename = "data/fine_pruning_model_for_anonymous_1_bd_net.h5"
poisoned_model_filename = "models/anonymous_1_bd_net.h5"

def random_pruning_model():
    # repaired_model_filename = "data/random_pruning_model_for_anonymous_1_bd_net.h5"
    # poisoned_model_filename = "models/anonymous_1_bd_net.h5"
    bd_model = keras.models.load_model(poisoned_model_filename)
    repaired_model = keras.models.load_model(repaired_model_filename)
    model = RandomPruningModel(poisoned=bd_model, repaired=repaired_model, N=1283)

    e = Evaluator(model=model, model_name='random_pruning')
    performance = e.evaluate()
    return performance

# def fine_pruning_model():
#     repaired_model_filename = "data/fine_pruning_model_G1.h5"
#     poisoned_model_filename = "models/sunglasses_bd_net.h5"
#     bd_model = keras.models.load_model(poisoned_model_filename)
#     repaired_model = keras.models.load_model(repaired_model_filename)
#     model = FinePruningModel(poisoned=bd_model, repaired=repaired_model, N=1283)
#
#     e = Evaluator(model=model, model_name='fine_pruning')
#     performance = e.evaluate()
#     return performance

def auto_encoder_repaired_model():
    # repaired_model_filename = "data/random_pruning_model_for_anonymous_1_bd_net.h5"
    repaired_model = keras.models.load_model(repaired_model_filename)
    AE = AutoEncoder()
    model = AutoEncoderRepairedModel(AE,repaired_model)
    e = Evaluator(model = model, model_name='auto_encoder_repaired')
    performance = e.evaluate()
    return performance

def auto_encoder_bd_model():
    repaired_model = keras.models.load_model(poisoned_model_filename)
    AE = AutoEncoder()
    model = AutoEncoderRepairedModel(AE,repaired_model)
    e = Evaluator(model = model, model_name='auto_encoder_badnet')
    performance = e.evaluate()
    return performance

if __name__ == '__main__':
    result = [
        random_pruning_model().kv,
        # fine_pruning_model().kv,
        auto_encoder_repaired_model().kv,
        auto_encoder_bd_model().kv
    ]
    df = pd.DataFrame(data=result)
    report = 'benchmark-report.csv'
    df.to_csv(report, index=False)
    print("benchmark report is saved to file {}".format(report))

    viz = Visualizer(df.set_index('model_name').T)
    viz.saveToFile()