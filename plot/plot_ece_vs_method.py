from os import read
import matplotlib.pyplot as plt
import json
import math


DATASETS = ['conll2003', 'GUM', 'wikiann', 'wnut_17']
DATASETS = ['conll2003', 'GUM']
METHODS = ['immediate_noise', 'immediate_no_noise']
DATASET_COLOR = {'conll2003': '#481D24', 
                 'GUM': '#E9724C',
                 'ncbi_disease': 'red', 
                 'wikiann': '#C5283D', 
                 'wnut_17': '#255F85'}

X = [1, 2, 3, 4]

def read_ece():
    results_per_method_per_dataset = {m: {d:[] for d in DATASETS} for m in METHODS}

    for dataset in DATASETS:
        for method in METHODS:
            for num in X:
                with open(f'../test_run3/{method}_{num}hidden/{dataset}/all_results.json') as fh:
                    results = json.loads(fh.read())
                    results_per_method_per_dataset[method][dataset].append(results['eval_ece'])
    return results_per_method_per_dataset


r = read_ece()

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1, 1, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)

def method_label(m):
    return m.replace("_", " ").replace("immediate", "abstention")

for m in METHODS:
    for d in DATASETS:
        line, = plt.plot(X, r[m][d], color=DATASET_COLOR[d])
        line.set_label(f'{method_label(m)} {d}'.capitalize())
        tx = X[0] + .3
        ty = r[m][d][0]
        plt.text(
            X[-1],
            r[m][d][-1],
            " — " + line.get_label(),
            size="small",
            color=line.get_color(),
            ha="left",
            va="center",
        )
        
        # ax.text(
        #     tx,
        #     ty,
        #     " " + line.get_label(),
        #     family="Roboto Condensed",
        #     size="small",
        #     bbox=dict(facecolor="white", edgecolor="None", alpha=0.85),
        #     color=line.get_color(),
        #     ha="center",
        #     va="center",
        #     rotation=0.,
        # )

plt.title("Calibration / Methods and Classifier Size", fontname="DejaVu", fontweight="bold")
plt.xlabel("Classifier Size", fontdict={'size': 'large'})
plt.xticks([1., 2., 3., 4.])
plt.ylabel("Calibration (ECE)", fontdict={'size': 'large'})
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.tight_layout()


plt.savefig("img.png")
