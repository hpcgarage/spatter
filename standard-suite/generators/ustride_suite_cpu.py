#!/usr/bin/env python3
# This file is used to create ../basic-test/cpu-ustride.json
# Set `memory` to the amount of memory you would like each test
# to use, 2 GB for example.

# This file used to iterate over many vector lengths (V)
# You can add another loop to do that if you like, but we found
# experimentally that V=3 (2**3 = 8) is good for CPUs and
# V=8 (2**8=256) is good for GPUs

import json

memory = 2 * (1000**3) # 2GB
data = []
elem = 8 #bytes per element
for kernel in ['Scatter', 'Gather']:
    V = 3 # index length 2**3 = 8
    for STRIDE in range(8): #8
        config ={}
        delta = 2**V * 2**STRIDE
        config['pattern'] = 'UNIFORM:{}:{}:NR'.format(2**V, 2**STRIDE)
        config['kernel'] = kernel
        config['count'] = memory // (delta*elem)
        data.append(config)

print(json.dumps(data))
