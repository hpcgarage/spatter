#!/usr/bin/env python3
# See  ustride_suite_cpu.py for an explanation of `V` and `memory`
# This file generates ../basic-test/gpu-ustride.py
# TODO: Combine this file with ustride_suite_cpu.py

import json

memory = 8 * (1000**3) # 8GB
data = []
elem = 8
V=8
for kernel in ['Scatter', 'Gather']:
    for STRIDE in range(8):
        config ={}
        delta = 2**V * 2**STRIDE
        config['pattern'] = 'UNIFORM:{}:{}:NR'.format(2**V, 2**STRIDE)
        config['kernel'] = kernel
        config['count'] = memory // (delta*elem)
        config['local-work-size'] = 1024
        data.append(config)

print(json.dumps(data))
