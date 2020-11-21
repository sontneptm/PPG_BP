import numpy as np
import pandas as pd
import glob
import pandas as pd
from scipy.stats import norm, kurtosis
from collections import Counter

def read():
    whole_data = np.loadtxt("wave_data_jeong.csv",delimiter=',', dtype=np.float32)
    print(len(whole_data[0]))


if __name__ == "__main__":
    read()