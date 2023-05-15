import time
import matplotlib.pyplot as plt
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", type=str)
parser.add_argument("-p", type=int, default=300)
args = parser.parse_args()
while(1):
    with open(args.f) as f:
        data = f.readlines()
        dps = []
    for line in data:
        if "total_loss" not in line:
            continue
        start = line.find("total_loss")
        line = line[start:]
        dps.append(float(line.split(" ")[1]))
    dps = np.array(dps)
    dps_mean = []
    for i in range(len(dps)):
        dps_mean.append(np.mean(dps[i-args.p:i]))
    plt.clf()
    plt.plot([i for i in range(len(dps))], dps_mean)
    plt.draw()
    plt.pause(30)
