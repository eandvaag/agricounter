import math as m
import numpy as np

import matplotlib.pyplot as plt

from exp_runner import my_plot_colors



def f(x):
    return 1 / (1 + m.e ** ((-10) * (x - 0.80)))

def f2(x):
    return 1 / (1 + m.e ** ((-0.3) * (x - 80)))

x = np.arange(0.25, 1.00, 0.01)

px = 1 / plt.rcParams["figure.dpi"]

fig, ax = plt.subplots(figsize=(853*px, 658*px))

ax.plot(x, f(x), color=my_plot_colors[0])

ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=8)
ax.set_title("Quality Score Weighting Function", fontsize=20)
ax.set_xlabel("Confidence Score", fontsize=16)
ax.set_ylabel("Weight", fontsize=16)
plt.tight_layout()
plt.savefig("confidence_quality.svg") #png")