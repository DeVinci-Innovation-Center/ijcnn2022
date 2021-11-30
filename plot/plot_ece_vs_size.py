from os import read
import matplotlib.pyplot as plt
import json
import math


DATASETS = ['conll2003', 'GUM', 'wikiann', 'wnut_17']
DATASETS = ['ncbi_disease']
DATASET_COLOR = {'conll2003': '#481D24', 
                 'GUM': '#E9724C',
                 'ncbi_disease': 'red', 
                 'wikiann': '#C5283D', 
                 'wnut_17': '#255F85'}

X = [1, 2, 3, 4, 5, 6, 7, 8]

def read_ece():
    results_per_dataset = {d: [] for d in DATASETS}

    for num in X:
        for dataset in DATASETS:
            with open(f'../test/immediate_noise_{num}hidden/{dataset}/all_results.json') as fh:
                results = json.loads(fh.read())
                results_per_dataset[dataset].append(results['eval_ece'])
    return results_per_dataset


r = read_ece()

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1, 1, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)

for d in DATASETS:
    line, = plt.plot(X, r[d], color=DATASET_COLOR[d])
    line.set_label(d.capitalize())
    # plt.text(
    #     X[-1],
    #     r[d][-1],
    #     " — " + line.get_label(),
    #     size="small",
    #     color=line.get_color(),
    #     ha="left",
    #     va="center",
    # )
    tx = X[0] + .3
    ty = r[d][0]
    ax.text(
        tx,
        ty,
        " " + line.get_label(),
        family="Roboto Condensed",
        size="large",
        bbox=dict(facecolor="white", edgecolor="None", alpha=0.85),
        color=line.get_color(),
        ha="center",
        va="center",
        rotation=0.,
    )

plt.title("Calibration w.r.t Classifier Size", fontname="DejaVu", fontweight="bold", fontdict={"size": "20"})
plt.xlabel("Classifier Size", fontdict={'size': 'large'})
plt.xticks(X)
plt.ylabel("Calibration (ECE)", fontdict={'size': 'large'})
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.tight_layout()

# plt.vlines(9, 0, 0.2, linestyles="dashed", color="grey")
# ax.text(9.1, 0.185, "< Optimal Depth", color="#525252", size="large")


plt.savefig("img.png")
