import json
import sys
import numpy as np

FILENAME=sys.argv[1]

with open(FILENAME, 'r', encoding='utf-8') as f:
    stats = json.load(f)

aggregated_stats = {}
for sample in stats:
    for k,v in sample.items():
        if k not in aggregated_stats.keys():
            aggregated_stats[k]= []
        aggregated_stats[k].append(v) 




for k,v in aggregated_stats.items():
    if k == "E2E":
        print(np.mean(v))
        break
