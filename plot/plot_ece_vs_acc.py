from os import read
import matplotlib.pyplot as plt
import matplotlib
import json
import math
import random
import re


matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

DATASETS = ['conll2003', 'GUM', 'wikiann', 'wnut_17', 'ncbi_disease']

DATASET = 'ncbi_disease'

METHODS = ['immediate_noise_{layer}hidden', 'immediate_no_noise_{layer}hidden', 'combine_immediate_unfrozen_{layer}hidden_64width/'] 


METHODS = ['raw_{layer}hidden_64width', 'combine_avuc_{layer}hidden_64width', 'combine_immediate_unfrozen_{layer}hidden_64width', 'noise_{layer}hidden_64width', 'immediate_noise_{layer}hidden_64width'] 


DATASET_COLOR = {'conll2003': '#481D24', 
                 'GUM': '#E9724C',
                 'ncbi_disease': 'red', 
                 'wikiann': '#C5283D', 
                 'wnut_17': '#255F85'}

X = [1]

def read_ece():
    ece = []
    f1 = []
    labels = []
    
    for method in METHODS:
        for num in X:
            label = method.format(layer=num)
            with open(f'../save/test_save_pour_pas_suppr/{label}/{DATASET}/all_results.json') as fh:
                results = json.loads(fh.read())
                ece.append(results['eval_ece'])
                f1.append(results['eval_f1'])
                labels.append(label)
    return ece, f1, labels


x, y, labels = read_ece()

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1, 1, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)

def method_label(m):
    # m = m.replace("{layer}hidden", "")
    m = m.replace("_", " ").replace("immediate", "Abstention")
    m = m.replace("noise", "Noise")
    m = m.replace('combine', '').replace('unfrozen', '').replace('raw', 'Baseline').replace('avuc', 'AVUC')
    m = m.replace("1hidden", '')
    m = m.replace("64width", '')
    m = m.replace('Abstention Noise', 'Abstention + Noise')
    return m

print(x)
print(y)

def get_color(i):
    colors = ['#a32618','#824314','#589113','#139180', '#171391', '#911384']
    return colors[i%len(colors)]
    r = lambda: random.randint(0,255)
    return '#%02X%02X%02X' % (r(),r(),r())


for i in range(len(x)):
    plt.scatter(x[i], y[i], marker='+', label=method_label(labels[i]), color=get_color(i), linewidths=2)
ax.legend()

for m in range(len(x)):
    # line, = plt.plot(X, r[m][d], color=DATASET_COLOR[d])
    # line.set_label(f'{method_label(m)} {d}'.capitalize())
    tx = x[m]
    ty = y[m]
    # plt.text(
    #     X[-1],
    #     r[m][d][-1],
    #     " — " + line.get_label(),
    #     size="small",
    #     color=line.get_color(),
    #     ha="left",
    #     va="center",
    # )
    
    # ax.text(
    #     tx,
    #     ty,
    #     " " + method_label(labels[m]),
    #     family="Roboto Condensed",
    #     size="small",
    #     bbox=dict(facecolor="white", edgecolor="None", alpha=0.25),
    #     color="red",
    #     ha="center",
    #     va="center",
    #     rotation=0.,
    # )

# plt.title("Calibration / Methods and Classifier Size", fontname="DejaVu", fontweight="bold")
# plt.title("Calibration / Methods and Classifier Size", fontname="DejaVu", fontweight="bold", fontdict={"size": "20"})
plt.xlabel("Calibration (ECE)", fontdict={'size': 'large'})

plt.ylabel("Performance (F1)", fontdict={'size': 'large'})
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.tight_layout()

rx = 5*(max(x) - min(x)) / 100
ry = 5*(max(y) - min(y)) / 100

plt.xlim(min(x)-rx, max(x)+rx)
plt.ylim(min(y)-ry, max(y)+ry)


fig.set_size_inches(w=3.57, h=3.57)
plt.savefig("img.pgf", bbox_inches="tight")
plt.savefig("img.png")
