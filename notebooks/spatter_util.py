# Add a new platform by adding it to one of
# the two following dicts. Also, make sure to
# add a color for it.

# You will also need to add your experiment to the list if
# it is not already there.

GPU_NAMES = {
        'gv100':'V100',
        'k40':'K40',
        'p100':'P100',
        'titan':'Titan XP',
        'customcgpu':'CustomGPU',
       }

CPU_NAMES = {
        'clx':'Cascade Lake',
        'bdw':'Broadwell',
        'skx':'Skylake',
        'tx2':'Thunder X2',
        'knl':'Knights Landing',
        'npl':'Naples',
        'customcpu':'CustomCPU',
       }

COLORS = {
        'gv100':'mediumvioletred',
        'k40':'blue',
        'p100':'black',
        'titan':'purple',
        'clx':'red',       #PLACEHOLDER
        'bdw':'#005596',
        'skx':'#26CAD3',
        'tx2':'orangered', #PLACEHOLDER
        'knl':'green',
        'npl':'purple',    #PLACEHOLDER
        'hsw':'#64d1a2',
        'customcpu':'red',
        'customgpu':'red',
       }

EXPERIMENTS = {'ustride':'Uniform Stride', 'stream': 'Stream', 'nekbone':'Nekbone', 'lulesh':'LULESH', 'amg':'AMG', 'pennant':'PENNANT'}

#################################################################
# NO EDITING IS REQUIRED BEYOND THIS POINT TO ADD NEW PLATFORMS #
#################################################################

import numpy as np
import pandas as pd
import ntpath #get_arch
import os     #get_arch
import math   #get_gap
from io import StringIO

ALLARCH = list(GPU_NAMES.keys()) + list(CPU_NAMES.keys())
ALLNAMES = {**GPU_NAMES, **CPU_NAMES}

# These functions have constant value for a file so need not be vectorized
def get_arch(str):
    arch = ntpath.basename(os.path.splitext(str)[0])
    arch = arch[arch.find("_")+1:]
    if arch not in ALLARCH:
        raise ValueError(f"Parsed arch as '{arch}' which is not in `ALLARCH`.")
    return arch

def get_experiment(str):
    exper = ntpath.basename(os.path.splitext(str)[0])
    exper = exper[:exper.find("_")]
    if exper not in EXPERIMENTS.keys():
        raise ValueError("Failed to parse experiment from \"" + str + "\"")
    return exper

# The following functions will be vectorized so that they may operate on lists
def get_archtype(str):
    str = str.lower()
    if str in CPU_NAMES:
        return 'CPU'
    elif str in GPU_NAMES:
        return 'GPU'
    else:
        raise ValueError("Unable to determine archtype for \"" + str + "\"")
        return 'UNKNOWN'

def get_gap(pattern, take_log):
    if len(pattern) < 2:
        return 0
    if take_log:
        if pattern[1] - pattern[0] <= 0:
            return 0
        return int(math.log2(pattern[1] - pattern[0]))
    else:
        return pattern[1] - pattern[0]

def get_pat_len(pattern):
    return len(pattern)

def get_color(arch):
    return COLORS[arch]

def get_pretty(arch):
    return ALLNAMES[arch]

# Vectorize previous five functions
get_archtype = np.vectorize(get_archtype)
get_gap      = np.vectorize(get_gap)
get_pat_len  = np.vectorize(get_pat_len)
get_color    = np.vectorize(get_color)
get_pretty   = np.vectorize(get_pretty)

# This is the main driver of this functionality. It is probably the only method that
# should be called from outside.

def file2df(filename, restrict_pat_len=0, archtype=None):
    #print("Parsing: " + filename)

    with open(filename, 'r') as file:
        contents = file.read()

    # Read run configurations
    start = contents.find("[", contents.find("Run Configurations"))
    end = contents.find("config")
    config = pd.DataFrame(eval(contents[start:end]))

    # Read data
    stats = contents.find("Min")
    data = pd.read_csv(StringIO(contents[end:stats]), delim_whitespace=True)

    # Join config with data
    table = data.join(config, on='config', how='inner')

    # Add additional columns to the data
    table['arch']       = get_arch(filename)
    table['archtype']   = get_archtype(table['arch'])
    table['experiment'] = get_experiment(filename)
    table['gap']        = get_gap(table['pattern'], True)
    table['delta']      = get_gap(table['pattern'], False)
    table['pat_len']    = get_pat_len(table['pattern'])
    table['color']      = get_color(table['arch'])
    table['pretty']     = get_pretty(table['arch'])

    # Additional options
    if restrict_pat_len > 0:
        table = table[table['pat_len'] == 256]

    return table
