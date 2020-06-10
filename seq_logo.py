import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import logomaker as lm
aas = 'ACDEFGHIKLMNPQRSTVWY'

def _visualizeScores(array, filename, ips_positions):
    arr = np.nan_to_num(array,copy=False)
    vals = arr
    df = pd.DataFrame(vals)
    df.columns = list(aas)
    max_scr = np.max(vals)
    min_scr = np.min(vals)

    logo = lm.Logo(df, figsize=(1+0.15*XXX, 2), color_scheme='skylign_protein')
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

    logo.ax.set_ylim(min_scr * 1.05, max_scr * 1.05)
    logo.ax.set_ylabel('contribution')

    ### code for highlighting ###
    cutoff = 0.65 # arbitrary value ; all scores above 'cutoff' % of the maximum score will be highlighted in red
    positions_to_highlight_red = [p for p in range(len(array)) if max(array[p]) > cutoff*max_scr]
    positions_to_highlight_yellow = ips_positions
    for p in positions_to_highlight_yellow:
        plt.axvspan(p - 0.5, p + 0.5, color='yellow', alpha=0.2, lw=0)
    for p in positions_to_highlight_red:
        plt.axvspan(p - 0.5, p + 0.5, color='red', alpha=0.2, lw=0)
    ### end code for highlighting ###

    plt.tight_layout()
    plt.savefig(filename)

### CODE TO TEST ###
XXX=200
array = np.zeros((XXX,20))
for i in range(XXX):
    r = random.randint(0,19)
    v = random.randint(0,99)/99-0.5
    array[i][r] = v

ips_positions = [50,51,52,53,54,55,56,57,58,59,60]
filename = 'test.png'
_visualizeScores(array,filename, ips_positions)
### END CODE TO TEST ###
