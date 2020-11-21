from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def plot_ppg_data():
    whole_data = np.loadtxt("ppg_bp_filtered.csv",delimiter=',', dtype=np.float32)
    ppg_data = whole_data[:,2:]
    x_axis = range(len(ppg_data[0]))

    plt.plot(x_axis, ppg_data[13])
    plt.show()
    
def plot_ppg_data2():
    whole_data = np.loadtxt("wave_data_jeong.csv",delimiter=',', dtype=np.float32)
    #whole_data = pd.read_csv("wave_data_jeong1.csv",delimiter=',', dtype=np.float32)
    ppg_data = whole_data[:,:]
    x_axis = range(len(ppg_data[1]))

    plt.plot(x_axis, ppg_data[0])
    plt.show()

def plot_ppg_data3():
    whole_data = np.loadtxt("ppg_bp_encoded.csv",delimiter=',', dtype=np.float32)
    #whole_data = pd.read_csv("wave_data_jeong1.csv",delimiter=',', dtype=np.float32)
    ppg_data = whole_data[:,:]
    x_axis = range(len(ppg_data[1]))

    plt.plot(x_axis, ppg_data[0])
    plt.show()

if __name__ == "__main__":
    plot_ppg_data()
    plot_ppg_data2()