#!/bin/bash

rm -rf perf_*

python test.py

if ls -d perf_* &> /dev/null; then
    chmod -R 777 perf_*
else
    echo "Warning: No perf_* files/directories found after running test.py" >&2
fi