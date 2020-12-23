import numpy as np
import h5py
import keras


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255


def pruning(p, weights):
    thresh = np.percentile(weights[-1], p)
    super_threshold_indices = weights[-1] < thresh
    weights[-1][super_threshold_indices] = 0
    return weights

def apply_pruning_to_dense(layer):
    if layer.name in ['fc_2']:
        layer.set_weights(pruning(percentile, layer.get_weights()))
    return layer


clean_test_filename = 'data/clean_test_data.h5'
# poisoned_data_filename = 'data/sunglasses_poisoned_data.h5'
poisoned_data_filename = 'data/anonymous_1_poisoned_data.h5'


"""
    Model Requirement:
        model should have a method named "predict", which returns the predict class (NOT probabilities of all classes)
    Usage: 
        e = Evaluator(model)
        e.evaluate()
"""


class EvaluateResult(object):

    def __init__(self, **kwargs):
        self.kv = {}
        for k, v in kwargs.items():
            self.kv[k] = v

class Evaluator:

    def __init__(self, model_path=None, model = None, model_name='default'):
        if model:
            self.model = model
        else:
            self.model = keras.models.load_model(model_path)
        self.model_name = model_name

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

        return EvaluateResult(
            model_name=self.model_name,
            clean_acc=accuracy,
            attack_succ_rate=attack_success_rate,
            trigger_detect_rate=trigger_detection_rate
        )

class Visualizer:
    def __init__(self, df):
        self.df = df

    def saveToFile(self):
        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        self.df = (self.df * 100).round(2)
        n = len(self.df)
        models = self.df.columns

        width = .35
        x = np.arange(n)
        fig, ax = plt.subplots()


        for i, model_name in enumerate(models):
            autolabel(ax.bar(x * 0.8 - width / 2 + i * width/2, self.df[model_name], width/2, label=model_name))

        ax.set_ylabel('Percent')
        ax.set_xticks(x * 0.8)
        ax.set_xticklabels(np.array(list(self.df.index)))
        ax.legend(models, loc="lower center", bbox_to_anchor=(0.5, -0.3))

        fig.tight_layout()
        plt.show()