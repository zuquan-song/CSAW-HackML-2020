import matplotlib.pyplot as plt

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