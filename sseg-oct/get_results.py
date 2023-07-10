import pandas as pd
import pickle, os
import numpy as np

models = ['base', 'unet', 'linknet', 'pspnet', 'pan', 'attnunet', 'segmenter', 'deformunet']
suffixes = ['', '-aug', '-adv', '-aug-adv']
colmap = dict(zip(suffixes, [('no-adv', 'no-aug'), ('no-adv', 'aug'), ('adv', 'no-aug'), ('adv', 'aug')]))

mean = pd.DataFrame(
    columns=pd.MultiIndex.from_product([['no-adv', 'adv'], ['no-aug', 'aug'], ['fscore', 'iou']]),
    index=models
)
std = pd.DataFrame(
    columns=pd.MultiIndex.from_product([['no-adv', 'adv'], ['no-aug', 'aug'], ['fscore', 'iou']]),
    index=models
)



for suffix in suffixes:
    for model in models:
        results_path = f'../results/{model}{suffix}/fold-results.pickle'
        if os.path.exists(results_path):
            with open(results_path, 'rb') as file:
                results = pickle.load(file)
            _, metrics = zip(*results)
            fscore, iou = zip(*[(metric.values['fscore']/metric.n, metric.values['iou']/metric.n) for metric in metrics])
            mean.loc[model][colmap[suffix]] = tuple(map(lambda x: round(np.mean(x), 2), [fscore, iou]))
            std.loc[model][colmap[suffix]] = tuple(map(lambda x: round(np.std(x), 2), [fscore, iou]))

print(mean)
print(std)
mean.to_excel('../results/kfold-means.xlsx', header=True, index=True)
mean.to_latex('../results/kfold-means.tex', header=True, index=True)
std.to_excel('../results/kfold-stds.xlsx', header=True, index=True)
std.to_latex('../results/kfold-stds.tex', header=True, index=True)

