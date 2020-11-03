import numpy as np
import pandas as pd
import glob
import pandas as pd

def text_to_csv():
    ppg_data_list = glob.glob('0_subject/*.txt')
    bp_data = np.loadtxt("bp_data.csv",delimiter=',', dtype=np.float32)

    file = open('ppg_bp.csv','a')
    
    ppg_bp_list = []

    print("loading text...")
    for p in ppg_data_list:

        label = p[p.index('t')+2:p.index('_',2)]
        subject = None
        for d in bp_data:
            if d[0] == float(label) :
                subject = d
        
        ppg = np.genfromtxt(p, dtype=None)
        if len(ppg) >2100:
            tmp = ppg[:2100]
            rtn_str = []
            rtn_str.append(subject[1])
            rtn_str.append(subject[2])
            for d in tmp:
                rtn_str.append(d)
            rtn = str(rtn_str)[1:-1]
            ppg = ppg[2100:]

        rtn_str = []
        rtn_str.append(subject[1])
        rtn_str.append(subject[2])
        for d in ppg:
            rtn_str.append(d)
        rtn = str(rtn_str)[1:-1]
        file.write(str.format(rtn) + "\n")
    print("done")

def ma_filter():
    file = open('ppg_bp_filtered.csv','a')
    whole_data = np.loadtxt("ppg_bp.csv",delimiter=',', dtype=np.float32)
    bp_data = whole_data[:,:2]
    ppg_data = whole_data[:,2:]

    for i in range(len(ppg_data)) :
        ppg_pd = pd.Series(ppg_data[i])
        ppg_data[i] = ppg_pd.rolling(window=10, min_periods=1).mean()

        tmp_list = []
        tmp_list.append(bp_data[i].tolist()[0])
        tmp_list.append(bp_data[i].tolist()[1])
        for d in ppg_data[i]:
            tmp_list.append(d)
        rtn = str(tmp_list)[1:-1]
        file.write(str.format(rtn) + "\n")

if __name__ == '__main__':
    ma_filter()