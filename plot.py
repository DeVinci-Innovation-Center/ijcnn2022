import matplotlib.pyplot as plt
import json

def r(n):
    with open(f'save/custom_grid/{n}_result/all_results.json') as fh:
        return json.loads(fh.read())["eval_ece"]


dta = [0.3, 0.35, 0.4, 0.45]

plt.plot(dta, [r(i) for i in dta])
plt.title("ECE by scaler")
plt.xlabel("Scale")
plt.ylabel("Eval ECE Calibration")
plt.savefig("img.png")