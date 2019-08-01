import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pandas.compat import StringIO
import sys
import re
import os
import ntpath

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

colors = {'on':'#26CAD3', 'off':'black'}

def prefetch(name):
    n = ntpath.basename(os.path.splitext(name)[0])
    n = n[n.find("_")+1:n.rfind("_")]
    if n.find("on") != -1:
        return "on"
    elif n.find("off") != -1:
        return "off"
    raise Exception("could not determine prefetch")

def get_arch(name):
    n = ntpath.basename(os.path.splitext(name)[0])
    n = n[n.rfind("_")+1:]
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
        tmp['prefetch'] = prefetch(f)
        tmp['norm_local'] = tmp['bw(MB/s)'] / max(tmp['bw(MB/s)'])
        dfs.append(tmp)
    df = pd.concat(dfs)

    df['norm_global'] = df['bw(MB/s)'] / max(df['bw(MB/s)'])
    df['bw(GB/s)'] = df['bw(MB/s)'] / 1000

    all_arch = ""
    for key, _ in df.groupby(['arch']):
        all_arch = all_arch + key


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
    print("Making plot 1")
    fix, ax = plt.subplots()
    #ax = plt.subplot(1, 2, 1) #plot 1

    for key, grp in df.groupby(['prefetch']):
        ax = grp.plot(ax=ax, kind='line', x='config', y='bw(GB/s)', label=key, color=colors[key])
        print(key)

    ax.set_xlabel("Stride (Doubles)")
    ax.set_ylabel("Bandwidth (GB/s)")

    ax.get_legend().remove()

    ax2 = ax.twinx()
    for key, grp in df.groupby(['prefetch']):
        ax2 = grp.plot(ax=ax2, kind='line', x='config', y='norm_global', label=key, color=colors[key], linewidth=4)
        print(key)


    MODE="normal"
    print(f"Mode is {MODE}")


    ax2.set_ylabel("Normalized bandwidth")
    ax2.get_legend().remove()

    if MODE == "opt":
        ax2.axhline(y=1, linestyle=":", color="black")
        ax2.axhline(y=.5, linestyle=":", color="black", xmin=1/7)
        ax2.axhline(y=.25, linestyle=":", color="black", xmin=2/7)
        ax2.axhline(y=.125, linestyle=":", color="black", xmin=3/7)
        ax2.axhline(y=.0625, linestyle=":", color="black", xmin=4/7)
    else:
        ax2.axhline(y=1, linestyle=":", color="black")
        ax2.axhline(y=.5, linestyle=":", color="black")
        ax2.axhline(y=.25, linestyle=":", color="black")
        ax2.axhline(y=.125, linestyle=":", color="black")
        ax2.axhline(y=.0625, linestyle=":", color="black")

    plt.yticks([.0625, .125, .25, .5, 1], ['1/16', '1/8', '1/4', '1/2', '1'])
    ax2.set_xticklabels([])
   # ax.set_xticklabels([7, 7, 7, 7, 7, 7, 7, 7])
    #def format_func(value, tick_number):
    #    return r"$2^{}$".format(value)
    #ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticklabels(["$2^{{{}}}$".format(x) for x in range(0,8)])
    #ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    #ax.ticklabel_format(axis='x', useMathText=True)
    
    ax.tick_params(axis=u'y', which=u'both',length=0)
    ax2.tick_params(axis=u'both', which=u'both',length=0)

    if MODE == "opt":
        for a in [ax, ax2]:
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
            a.spines["bottom"].set_visible(False)
            a.spines["left"].set_visible(False)
 
    plt.legend(loc='best', title='Prefetch')

    # Change figure size
    fig = plt.gcf()
    fig.set_size_inches(6, 6)

    outname = "prefetch_{}_{}.png".format(all_arch, MODE)
    plt.savefig(outname, transparent=True, bbox_inches='tight')
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticklabels(["$2^{{{}}}$".format(x) for x in range(0,8)])
    print("Saved plot to {}".format(outname))

    handles,labels = ax.get_legend_handles_labels()
    handles = [handles[1], handles[0]]
    labels = [labels[1], labels[0]]

    plt.clf()
    exit(0)

    #Plot against a local max
    fig, ax = plt.subplots()
    for key, grp in df.groupby(['arch']):
        ax = grp.plot(ax=ax, kind='line', x='gap', y='norm_local', label=key)
        print(key)

    plt.legend(loc='best', title='log2(gap)')

    outname = "ustride_local.png"
    plt.savefig(outname)
    plt.clf()
