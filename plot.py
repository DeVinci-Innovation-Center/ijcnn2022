import matplotlib.pyplot as plt
import json

def r(n):
    with open(f'save/custom_grid/{n}_result/all_results.json') as fh:
        return json.loads(fh.read())["eval_ece"]


dta = [0.01, 0.025, 0.05, 0.075, 0.1]

plt.plot(dta, [r(i) for i in dta])
plt.title("ECE by scaler")
plt.xlabel("Scale")
plt.ylabel("Eval ECE Calibration")
plt.savefig("img.png")