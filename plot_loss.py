from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

pinwheel_totol_loss=np.load("experiments/cnf/toy/figs_penalty_pinwheel/total_loss.npy")
spirals_totol_loss=np.load("experiments/cnf/toy/figs_penalty_2spirals/total_loss.npy")
moons_totol_loss=np.load("experiments/cnf/toy/figs_penalty_moon/total_loss.npy")
gaussians8_totol_loss=np.load("experiments/cnf/toy/figs_penalty_8gaussians/total_loss.npy")

plt.plot(pinwheel_totol_loss[10:],label="pinwheel")
plt.plot(spirals_totol_loss[10:],label="2spirals")
plt.plot(moons_totol_loss[10:],label="moons")
plt.plot(gaussians8_totol_loss[10:],label="8gaussians")
plt.xlabel("iteration")
plt.ylabel("Loss")
plt.legend()
plt.show()