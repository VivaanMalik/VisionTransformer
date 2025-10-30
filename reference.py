import json
import numpy as np

def shape(arr):
    if isinstance(arr, list):
        return (len(arr), *shape(arr[0])) if arr else ()
    return ()

data = json.load(open("parameters/weights.json"))
for k,v in data.items():
    print(k, shape(v))