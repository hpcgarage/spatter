import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.compat import StringIO
import matplotlib.ticker as ticker
import sys
import re
import os
import ntpath

colors={'tx2':'orange', 'skx':'#26CAD3', 'bdw':'#005596', 'hsw':'#64d1a2'}

def file_to_df(filename):
    with open(filename, 'r') as file:
        contents = file.read()

    # Read run configurations
    start = contents.find("[", contents.find("Run Configurations"))
    end = contents.find("config")
    config = pd.DataFrame(eval(contents[start:end]))
    #config['config'] = [i for i in range(config.shape[0])]

    # Read data
    data = pd.read_csv(StringIO(contents[end:]), delim_whitespace=True)

    # Join tables and return 
    return data.join(config, on='config', how='inner')

def _is_nr(str):
    return str.find("NR") != -1
is_nr = np.vectorize(_is_nr)

def _gap(arr):
    if len(arr) < 2:
        raise Exception('length 0 or 1 pattern')
    return arr[1]
gap = np.vectorize(_gap)

def _pct(name):
    return int(re.findall('\d+', name)[0])
pct = np.vectorize(_pct)

def get_arch(name):
    n = ntpath.basename(os.path.splitext(name)[0])
    n = n[n.find("_")+1:]
    return n


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 {} input.dat".format(sys.argv[0]))
        exit(1)

    #Read files
    dfs = []
    for f in sys.argv[1:]:
        tmp = file_to_df(f)
        tmp['arch'] = get_arch(f)
        tmp['gap'] = gap(tmp['name'])
        tmp['norm_local'] = tmp['bw(MB/s)'] / max(tmp['bw(MB/s)'])
        dfs.append(tmp)
    df = pd.concat(dfs)

    df['norm_global'] = df['bw(MB/s)'] / max(df['bw(MB/s)'])

    SMALL_SIZE = 15
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    #Plot against global max
    fig, ax = plt.subplots()
    for key, grp in df.groupby(['arch']):
        ax = grp.plot(ax=ax, kind='line', x='config', y='bw(MB/s)', label=key, color=colors[key], linewidth=4)
        print(key)

    #ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("Log(Bandwidth)")
    ax.set_xlabel("Stride (Doubles)")

    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticklabels(["$2^{{{}}}$".format(x) for x in range(0,8)])

    handles,labels = ax.get_legend_handles_labels()
    handles = [handles[2], handles[0], handles[1], handles[3]]
    labels = [labels[2], labels[0], labels[1], labels[3]]

    plt.legend(handles, labels, loc='best', title='')
    #plt.rcParams.update({'font.size': 28})

    fig.set_size_inches(6,6)
    outname = "ustride_actual.png"
    plt.savefig(outname, transparent=True)
    print(f"wrote to {outname}")
    plt.clf()
    exit(0)

#########################
    #Plot against a local max
    fig, ax = plt.subplots()
    for key, grp in df.groupby(['arch']):
        ax = grp.plot(ax=ax, kind='line', x='gap', y='norm_local', label=key)
        print(key)

    plt.legend(loc='best', title='log2(gap)')

    outname = "ustride_tmp.png"
    plt.savefig(outname)
    print(f"wrote to {outname}")
    plt.clf()
