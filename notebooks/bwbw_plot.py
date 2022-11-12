#!/usr/bin/env python3
import spatter_util as su
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
#from colour import Color
#from cycler import cycler
#import model_bw as model

if (len(sys.argv) > 1):
    _kernel = sys.argv[1]
    if (_kernel != 'Gather' and _kernel != 'Scatter'):
        print(f"Unknown kernel type: {_kernel}.")
        _kernel = 'Gather'

if (_kernel == 'Gather'):
    EXPER     = {'pennant': [2, 5, 14,10]} # Gather
    KERNEL    = 'Gather'
if (_kernel == 'Scatter'):
    EXPER     = {'lulesh':[0,3]} # Scatter
    KERNEL    = 'Scatter'

######################### Script Params ###########################
PLATFORMS = ['bdw', 'npl', 'tx2', 'titan', 'p100', 'gv100', 'clx', 'skx']
STRIDES   = [0, 4]
#EXPER     = {'pennant': [2, 5, 14,10]} # Gather
#KERNEL    = 'Gather'
#EXPER     = {'lulesh':[0,3]} # Scatter
#KERNEL    = 'Scatter'
###################################################################

############################ RENAME ###############################
# gather
rename = {'pennant-000': 'PENNANT-G2',
 'pennant-001': 'PENNANT-G3',
 'pennant-002': 'PENNANT-G12',
 'pennant-003': 'PENNANT-G0',
 'pennant-004': 'PENNANT-G1',
 'pennant-005': 'PENNANT-G7',
 'pennant-007': 'PENNANT-G11',
 'pennant-008': 'PENNANT-G10',
 'pennant-009': 'PENNANT-G5',
 'pennant-010': 'PENNANT-G15',
 'pennant-011': 'PENNANT-G13',
 'pennant-012': 'PENNANT-G14',
 'pennant-013': 'PENNANT-G6',
 'pennant-014': 'PENNANT-G8',
 'pennant-015': 'PENNANT-G4',
 'pennant-016': 'PENNANT-G9',
 'lulesh-001': 'LULESH-G2',
 'lulesh-004': 'LULESH-G4',
 'lulesh-005': 'LULESH-G6',
 'lulesh-006': 'LULESH-G3',
 'lulesh-008': 'LULESH-G1',
 'lulesh-009': 'LULESH-G7',
 'lulesh-010': 'LULESH-G0',
 'lulesh-011': 'LULESH-G5',
 'nekbone-000': 'NEKBONE-G0',
 'nekbone-001': 'NEKBONE-G2',
 'nekbone-002': 'NEKBONE-G1',
 'amg-000': 'AMG-G1',
 'amg-001': 'AMG-G0'}
# scatter
rename.update({'pennant-006': 'PENNANT-S0',
 'lulesh-000': 'LULESH-S3',
 'lulesh-002': 'LULESH-S0',
 'lulesh-003': 'LULESH-S1',
 'lulesh-007': 'LULESH-S2'})

###################################################################

## Globals
ORDER = []
min_ys = [np.inf]
max_ys = [0]

USTRIDE_LINES=1
USTRIDE_DASHES=.75
USTRIDE_POINTS=5
APP_LINES=1
APP_POINTS=5

USTRIDE_POINT_SIZE=10
USTRIDE_DASHED_LINE_WIDTH=.4
USTRIDE_SOLID_LINE_WIDTH=1
USTRIDE_COLOR='#424242'

APP_POINT_SIZE=13
APP_LINE_WIDTH=1.5
APP_COLOR='blue'

################################
# Detect if we are in notebook #
################################

def in_ipynb():
    try:
        cfg = get_ipython().config
        return True
    except:
        return False

IN_NOTEBOOK= in_ipynb()

if IN_NOTEBOOK:
    try:
        notebook_df
    except:
        print("Error: We are in a notebook but we couldn't find the variable containing the experiment data, `notebook_df`.")
        sys.exit(1)

    try:
        PLATFORMS = list(set(PLATFORMS) - set(notebook_remove))
    except:
        pass

    PLATFORMS.append(f'custom{notebook_system_type}')

def set_order(ord):
    global ORDER
    ORDER = ord.copy()

def reorder(a):
    global ORDER
    return [x for _, x in sorted(zip(ORDER,a), key=lambda pair: pair[0])]

# Read data from pickle file
data = pd.read_pickle("./pattern_results.pkl")

if (IN_NOTEBOOK):
    data = pd.concat([data, notebook_df], ignore_index=True)

# Subset data based on PLATFORMS and KERNEL param
def member(a):
   return a in PLATFORMS

member = np.vectorize(member)
data = data[member(list(data['arch']))]
data = data[data['kernel'] == KERNEL]

#models = model.get_models(data)

# We need to plot uniform stride results first
ustride = data[data['experiment'] == 'ustride']

# Initialize plot
fig, ax = plt.subplots()

# Get x's and store them to use for reordering other data. Then sort x's.
xs = ustride[ustride['gap'] == 0]['bw(MB/s)'].to_numpy()
set_order(xs)
xs = reorder(xs)

# Get architecture names
names = ustride[ustride['gap'] == 0]['arch'].to_numpy()
names = reorder(names)

# Plot uniform stride results
for g in STRIDES:
    ys = ustride[ustride['gap'] == g]['bw(MB/s)'].to_numpy()
    ys = reorder(ys)
    plt.scatter(xs, ys, s=USTRIDE_POINT_SIZE, color=USTRIDE_COLOR)

    # Used for setting bounds later
    min_ys = min([min_ys, *ys])
    max_ys = max([max_ys, *ys])

    # Add Line Label
    plt.text(xs[len(xs)-1]*1.1, ys[len(ys)-1]*1.1, "Stride-{}".format(2**g), rotation=45, color=USTRIDE_COLOR)

    # Plot lines connecting points
    for i in range(len(xs)):
        plt.plot(xs[i:i+2], ys[i:i+2], color=USTRIDE_COLOR, linestyle='solid', linewidth=USTRIDE_SOLID_LINE_WIDTH)
    # Plot line connecting stride MAX with stride 0
    if g == max(STRIDES):
        for i in range(len(xs)):
            line, = ax.plot([xs[i], xs[i]], [xs[i], ys[i]], color=USTRIDE_COLOR, linewidth=USTRIDE_DASHED_LINE_WIDTH)
            line.set_dashes([6,4])


# Plot application results
for ex in EXPER.keys():

    ## Generate color list for cycler
    #start_color = Color('#00adff')
    #end_color   = Color('#03034f')
    #n_colors    = len(EXPER[ex])
    ##color_list = list(start_color.range_to(end_color, len(EXPER.keys())))
    #color_list = list(start_color.range_to(end_color,n_colors))
    #color_list = [c.hex for c in color_list]
    #color_list = np.repeat(color_list, len(PLATFORMS)+2)
    #ax.set_prop_cycle(cycler('color', color_list))
    ##print(color_list)

    i = 0
    for con in EXPER[ex]:
        ys = data[(data['experiment'] == ex) & (data['config'] == con)]['bw(MB/s)'].to_numpy()
        if len(ys) == 0:
            print('Warning: ys is length 0. This likely means the experiment+config you are looking for does not use the kernel you specified. Skipping [{},{}].'.format(ex, con))
            continue
        ys = reorder(ys)


        plt.scatter(xs, ys, s=APP_POINT_SIZE, color=APP_COLOR)

        # Used for setting bounds later
        min_ys = min([min_ys, *ys])
        max_ys = max([max_ys, *ys])

        label = '{}-{:03d}'.format(su.EXPERIMENTS[ex], con).lower()
        con_remap = rename[label]

        for i in range(len(xs)):
            plt.plot(xs[i:i+2], ys[i:i+2], linestyle='solid', linewidth=APP_LINE_WIDTH, color=APP_COLOR)
        # Add Line Label
        plt.text(xs[len(xs)-1]*1.1, ys[len(ys)-1]*1.1, con_remap, rotation=45, color=APP_COLOR)
        i += 1

###########
# Visuals #
###########

plt.gcf().subplots_adjust(left=0.15)
ax.set_xscale('log')
ax.set_yscale('log')

#x and y axis limits
ulim = max(max(xs), max_ys) * 1.2
llim = min(min(xs), min_ys) * 0.7
plt.xlim((llim, ulim))
plt.ylim((llim, ulim))

# Remove top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add arch names
for i in range(len(xs)):
    plt.text(xs[i]*.95, xs[i], su.ALLNAMES[names[i]], fontsize=8, ha='right')

# Axis Labels
plt.xlabel('Stride-1 DRAM Bandwidth', labelpad=15)
plt.ylabel('Pattern Bandwidth', labelpad=15)

# Title
if len(EXPER) == 0:
    app_str = ''
elif (len(EXPER) == 1):
    app_str = ' vs {} Patterns'.format(su.EXPERIMENTS[list(EXPER.keys())[0]])
else:
    app_str = ' vs Application Patterns'

plt.suptitle('Uniform Stride Bandwidth{}'.format(app_str))

# Subtitle
plt.title('Selected {} Patterns'.format(KERNEL), fontsize=10, y=1.04)

# Ticks
plt.minorticks_off()

locs = [10**x for x in range(10) if 10**x >= llim and 10**x <= ulim]
labs_dict = {1:'1 MB/s', 10:'10 MB/s', 100:'100 MB/s',
        1000:'1 GB/s',10000:'10 GB/s', 100000:'100 GB/s',
        1000000:'1 TB/s', 10000000:'10 TB/s', 100000000:'100 TB/s',
        1000000000:'1 PB/s'}
labs = [labs_dict[i] for i in locs]
plt.xticks(locs, labs)
plt.yticks(locs, labs, rotation='vertical')

# Diagonal lines
x_space = np.linspace(llim, max_ys, 20)

plt.plot(x_space, x_space, linewidth=.5, color='#4f4f4f', linestyle='dashed')
plt.plot(x_space, x_space/2, linewidth=.5, color='#4f4f4f', linestyle='dashed')
plt.plot(x_space, x_space/4, linewidth=.5, color='#4f4f4f', linestyle='dashed')
plt.plot(x_space, x_space/8, linewidth=.5, color='#4f4f4f', linestyle='dashed')
plt.plot(x_space, x_space/16, linewidth=.5, color='#4f4f4f', linestyle='dashed')

# Text for diagonal line
plt.text(llim, llim*1.1, 'Full DRAM Bandwidth', rotation=45, fontsize=8)
plt.text(llim*2*.9, llim*1.0, '1/2 BW', rotation=45, fontsize=8)
plt.text(llim*4*.9, llim*1.0, '1/4', rotation=45, fontsize=8)
plt.text(llim*8*.9, llim*1.0, '1/8', rotation=45, fontsize=8)
plt.text(llim*16*.9, llim*1.0, '1/16', rotation=45, fontsize=8)

#############
# Save File #
#############

fig.tight_layout()
fig.set_size_inches(6, 6)

if not IN_NOTEBOOK:
    outname = "bwbw_{}.pdf".format(KERNEL)
    plt.savefig(outname)
    print('worte to {}'.format(outname))
