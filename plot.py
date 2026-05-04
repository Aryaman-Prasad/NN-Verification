import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV_FILE1 = "results/results_mnist_fc_256x2_relaxed_robust_50.csv"
CSV_FILE2 = "results/results_mnist_fc_256x2_relaxed_robust_mara_50.csv"
CSV_FILE3 = "results/results_mnist_fc_256x2_afzal_relaxed_robust_50.csv"
CSV_FILE4 = "results/results_mnist_fc_256x2_relaxed_robust_mara_v2_50.csv"
CSV_FILE5 = "results/results_mnist_fc_256x2_relaxed_robust_v2_50.csv"
CSV_FILE6 = "results/results_mnist_fc_256x2_relaxed_robust_mara_afzal_50.csv"
PLOT_FILE = "plots/mnist_fc_256x2_relaxed_robust_everything_50.png"

TIMEOUT = 60

csv1 = pd.read_csv(CSV_FILE1)
csv2 = pd.read_csv(CSV_FILE2)
csv3 = pd.read_csv(CSV_FILE3)
csv4 = pd.read_csv(CSV_FILE4)
csv5 = pd.read_csv(CSV_FILE5)
csv6 = pd.read_csv(CSV_FILE6)

times1 = csv1["time"]
times2 = csv2["time"]
times3 = csv3["time"]
times4 = csv4["time"]
times5 = csv5["time"]
times6 = csv6["time"]

times1 = times1.clip(lower=0, upper=TIMEOUT)
times2 = times2.clip(lower=0, upper=TIMEOUT)
times3 = times3.clip(lower=0, upper=TIMEOUT)
times4 = times4.clip(lower=0, upper=TIMEOUT)
times5 = times5.clip(lower=0, upper=TIMEOUT)
times6 = times6.clip(lower=0, upper=TIMEOUT)

times1.fillna(TIMEOUT, inplace = True)
times2.fillna(TIMEOUT, inplace = True)
times3.fillna(TIMEOUT, inplace = True)
times4.fillna(TIMEOUT, inplace = True)
times5.fillna(TIMEOUT, inplace = True)
times6.fillna(TIMEOUT, inplace = True)

times1_sort = sorted(times1)
times2_sort = sorted(times2)
times3_sort = sorted(times3)
times4_sort = sorted(times4)
times5_sort = sorted(times5)
times6_sort = sorted(times6)

n = len(times1)

plt.figure(figsize = (12, 8))
plt.plot(np.arange(1, n+1), times1_sort, color = "red", label = "abcrown - min/max encoding")
plt.plot(np.arange(1, n+1), times2_sort, color = "green", label = "marabou - min/max encoding")
plt.plot(np.arange(1, n+1), times3_sort, color = "blue", label = "abcrown - paper's encoding")
plt.plot(np.arange(1, n+1), times4_sort, color = "black", label = "marabou - our encoding")
plt.plot(np.arange(1, n+1), times5_sort, color = "purple", label = "abcrown - our encoding")
plt.plot(np.arange(1, n+1), times6_sort, color = "orange", label = "marabou - paper's encoding")
plt.legend()

plt.savefig(PLOT_FILE)


