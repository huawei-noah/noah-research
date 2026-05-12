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
    if k in ("TPOT", "TTFT", "TPOT_MLP", "E2E"):
        print(f"{k} : {np.mean(v)/1000:.2f} +- {np.std(v)/1000:.2f} (ms)")
    if k == "TPOT":
        v= 1_000_000/np.array(v)
        print(f"Decoding Speed: {np.mean(v):.2f} +- {np.std(v):.2f} (token/s)")
