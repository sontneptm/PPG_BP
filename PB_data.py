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

if __name__ == '__main__':
    text_to_csv()