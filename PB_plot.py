from matplotlib import pyplot as plt
import numpy as np

def plot_ppg_data():
    whole_data = np.loadtxt("ppg_bp_filtered.csv",delimiter=',', dtype=np.float32)
    ppg_data = whole_data[:,2:]
    x_axis = range(len(ppg_data[0]))

    plt.plot(x_axis, ppg_data[5])
    plt.show()
    
if __name__ == "__main__":
    plot_ppg_data()