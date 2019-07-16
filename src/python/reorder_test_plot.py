import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.compat import StringIO
import sys
import re

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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 {} input.dat".format(sys.argv[0]))
        exit(1)

    df = file_to_df(sys.argv[1])
    #df['gap'] = gap(df['pattern'])
    #df['pct'] = pct(df['name'])


    print("here")
    exit(0)
    print("here")
    fig, ax = plt.subplots()
    for key, grp in df.groupby(['pct']):
        ax = grp.plot(ax=ax, kind='line', x='gap', y='bandwidth(MB/s)', label=key)
        print(key)

    plt.legend(loc='best', title='% Reuse')
    plt.savefig('strides2_skx.png')
