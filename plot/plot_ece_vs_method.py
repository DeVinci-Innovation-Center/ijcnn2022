from os import read
import matplotlib.pyplot as plt
import matplotlib
import json
import math


matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


DATASETS = ['conll2003', 'GUM', 'wikiann', 'wnut_17', 'ncbi_disease']
DATASETS = ['ncbi_disease']
# METHODS = ['immediate_noise_{layer}hidden', 'immediate_no_noise_{layer}hidden', 'immediate_noise_{layer}hidden_64width'] # 'immediate_noise_{layer}hidden_256width'

METHODS = ['raw_{layer}hidden_64width', 'combine_immediate_unfrozen_{layer}hidden_64width', 'noise_{layer}hidden_64width', 'immediate_noise_{layer}hidden_64width'] 

DATASET_COLOR = {'conll2003': '#481D24', 
                 'GUM': '#E9724C',
                 'ncbi_disease': 'red', 
                 'wikiann': '#C5283D', 
                 'wnut_17': '#255F85'}

X = [1, 2, 3, 4, 5]

def read_ece():
    results_per_method_per_dataset = {m: {d: [] for d in DATASETS} for m in METHODS}

    

    for dataset in DATASETS:
        for method in METHODS:
            for num in X:
                label = method.format(layer=num)
                with open(f'../save/test_save_pour_pas_suppr/{label}/{dataset}/all_results.json') as fh:
                    results = json.loads(fh.read())
                    results_per_method_per_dataset[method][dataset].append(results['eval_ece'])
    return results_per_method_per_dataset

def method_label(m):
    # m = m.replace("{layer}hidden", "")
    m = m.replace("_", " ").replace("immediate", "Abstention")
    m = m.replace("noise", "Noise")
    m = m.replace('combine', '').replace('unfrozen', '').replace('raw', 'Baseline').replace('avuc', 'AVUC')
    m = m.replace("1hidden", '')
    m = m.replace("64width", '')
    m = m.replace('Abstention Noise', 'Abstention + Noise')
    m = m.replace('ncbi_disease', '').replace('{layer}hidden', '').replace('abstention', "Abstention")
    print(m)
    return m


r = read_ece()

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1, 1, 1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_visible(False)


colors = ['#a32618','#824314','#589113','#139180', '#171391', '#911384']
i = 0

for m in METHODS:
    for d in DATASETS:
        line, = plt.plot(X, r[m][d], color=colors[i%len(colors)])
        line.set_label(f'{method_label(m)}')
        i+=1


        # tx = X[0] + .3
        # ty = r[m][d][0]
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
        #     " " + line.get_label(),
        #     family="Roboto Condensed",
        #     size="small",
        #     bbox=dict(facecolor="white", edgecolor="None", alpha=0.85),
        #     color=line.get_color(),
        #     ha="center",
        #     va="center",
        #     rotation=0.,
        # )

ax.legend()

# plt.title("Calibration / Methods and Classifier Size", fontname="DejaVu", fontweight="bold")
# plt.title("Calibration / Methods and Classifier Size", fontname="DejaVu", fontweight="bold", fontdict={"size": "20"})
plt.xlabel("Classifier Layers", fontdict={'size': 'large'})
plt.xticks(X)
plt.ylabel("Calibration (ECE)", fontdict={'size': 'large'})
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.tight_layout()



# fig.set_size_inches(w=3.57, h=3.57)
plt.savefig("img.pgf", bbox_inches="tight")
plt.savefig("img.png", bbox_inches="tight")