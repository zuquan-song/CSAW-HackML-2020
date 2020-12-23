import pandas as pd
from rule_based_model_eval import RuleBasedModel
from auto_encoder_utils import *
from utils import *
from visualizer import Visualizer

random_pruning_repaired_model_filename = "fixed_models/random_pruning_model_for_anonymous_1_bd_net.h5"
fine_pruning_repaired_model_filename = "fixed_models/fine_pruning_model_for_anonymous_1_bd_net.h5"

# random_pruning_repaired_model_filename = "fixed_models/random_pruning_model_for_sunglasses_bd_net.h5"
# fine_pruning_repaired_model_filename = "fixed_models/fine_pruning_model_for_sunglasses_bd_net.h5"

poisoned_model_filename = "models/anonymous_1_bd_net.h5"

evaluate_clean_file = 'data/clean_test_data.h5'
evaluate_poisoned_file = 'data/anonymous_1_poisoned_data.h5'
# evaluate_poisoned_file = 'data/Multi-trigger Multi-target/sunglasses_poisoned_data.h5'

def random_pruning_rule_based_repaired_model():
    bd_model = keras.models.load_model(poisoned_model_filename)
    repaired_model = keras.models.load_model(random_pruning_repaired_model_filename)
    model = RuleBasedModel(poisoned=bd_model, repaired=repaired_model, N=1283)

    e = Evaluator(model = model,
                  model_name='random_pruning_rule_based_repaired_model',
                  clean_file=evaluate_clean_file,
                  poisoned_file=evaluate_poisoned_file
                  )
    performance = e.evaluate()

    return performance


def random_pruning_based_auto_encoder_repaired_model():
    repaired_model = keras.models.load_model(random_pruning_repaired_model_filename)
    AE = AutoEncoder()
    model = AutoEncoderRepairedModel(AE,repaired_model)
    e = Evaluator(model = model,
                  model_name='random_pruning_based_auto_encoder_repaired_model',
                  clean_file=evaluate_clean_file,
                  poisoned_file=evaluate_poisoned_file
                  )
    performance = e.evaluate()
    return performance

def random_pruning_based_auto_encoder_bd_model():
    repaired_model = keras.models.load_model(poisoned_model_filename)
    AE = AutoEncoder()
    model = AutoEncoderRepairedModel(AE,repaired_model)
    e = Evaluator(model = model,
                  model_name='random_pruning_based_auto_encoder_bd_model',
                  clean_file=evaluate_clean_file,
                  poisoned_file=evaluate_poisoned_file
                  )
    performance = e.evaluate()
    return performance


def fine_pruning_rule_based_repaired_model():
    bd_model = keras.models.load_model(poisoned_model_filename)
    repaired_model = keras.models.load_model(fine_pruning_repaired_model_filename)
    model = RuleBasedModel(poisoned=bd_model, repaired=repaired_model, N=1283)

    e = Evaluator(model = model,
                  model_name='fine_pruning_rule_based_repaired_model',
                  clean_file=evaluate_clean_file,
                  poisoned_file=evaluate_poisoned_file
                  )
    performance = e.evaluate()
    return performance

def fine_pruning_based_auto_encoder_repaired_model():
    repaired_model = keras.models.load_model(fine_pruning_repaired_model_filename)
    AE = AutoEncoder()
    model = AutoEncoderRepairedModel(AE,repaired_model)
    e = Evaluator(model = model,
                  model_name='random_pruning_based_auto_encoder_repaired_model',
                  clean_file=evaluate_clean_file,
                  poisoned_file=evaluate_poisoned_file
                  )
    performance = e.evaluate()
    return performance

def fine_pruning_based_auto_encoder_bd_model():
    repaired_model = keras.models.load_model(fine_pruning_repaired_model_filename)
    AE = AutoEncoder()
    model = AutoEncoderRepairedModel(AE,repaired_model)
    e = Evaluator(model = model,
                  model_name='random_pruning_based_auto_encoder_bd_model',
                  clean_file=evaluate_clean_file,
                  poisoned_file=evaluate_poisoned_file
                  )
    performance = e.evaluate()
    return performance

if __name__ == '__main__':
    result = [
        random_pruning_rule_based_repaired_model().kv,
        random_pruning_based_auto_encoder_repaired_model().kv,
        random_pruning_based_auto_encoder_bd_model().kv,
        fine_pruning_rule_based_repaired_model().kv,
        fine_pruning_based_auto_encoder_repaired_model().kv,
        fine_pruning_based_auto_encoder_bd_model().kv,
    ]
    df = pd.DataFrame(data=result)
    report = 'benchmark-report.csv'
    df.to_csv(report, index=False)
    print("benchmark report is saved to file {}".format(report))

    viz = Visualizer(df.set_index('model_name').T)
    viz.saveToFile()