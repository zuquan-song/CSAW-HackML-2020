import pandas as pd
from fine_pruning_model import FinePruningModel
from random_pruning_model import RandomPruningModel
from auto_encoder_utils import *
from utils import *
import matplotlib.pyplot as plt

def random_pruning_model():
    repaired_model_filename = "data/random_pruning_model_G1.h5"
    poisoned_model_filename = "models/sunglasses_bd_net.h5"
    bd_model = keras.models.load_model(poisoned_model_filename)
    repaired_model = keras.models.load_model(repaired_model_filename)
    model = RandomPruningModel(poisoned=bd_model, repaired=repaired_model, N=1283)

    e = Evaluator(model=model, model_name='random_pruning')
    performance = e.evaluate()
    return performance

def fine_pruning_model():
    repaired_model_filename = "data/fine_pruning_model_G1.h5"
    poisoned_model_filename = "models/sunglasses_bd_net.h5"
    bd_model = keras.models.load_model(poisoned_model_filename)
    repaired_model = keras.models.load_model(repaired_model_filename)
    model = FinePruningModel(poisoned=bd_model, repaired=repaired_model, N=1283)

    e = Evaluator(model=model, model_name='fine_pruning')
    performance = e.evaluate()
    return performance

def auto_encoder_repaired_model():
    repaired_model_filename = "fixed_model/pruning.h5"
    repaired_model = keras.models.load_model(repaired_model_filename)
    AE = AutoEncoder()
    model = AutoEncoderRepairedModel(AE,repaired_model)
    e = Evaluator(model = model, model_name='auto_encoder_repaired')
    performance = e.evaluate()
    return performance

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
        ax.set_xticklabels(list(df.index))
        ax.legend(models, loc="lower center", bbox_to_anchor=(0.5, -0.3))

        fig.tight_layout()
        plt.show()

if __name__ == '__main__':
    result = [
        random_pruning_model().kv,
        fine_pruning_model().kv,
        auto_encoder_repaired_model().kv
    ]
    df = pd.DataFrame(data=result)
    report = 'benchmark-report.csv'
    df.to_csv(report, index=False)
    print("benchmark report is saved to file {}".format(report))

    viz = Visualizer(df.set_index('model_name').T)
    viz.saveToFile()