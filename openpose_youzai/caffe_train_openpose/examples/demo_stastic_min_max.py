from __future__ import division
import sys, os


def main():
    lines = []
    with open(sys.argv[1]) as f:
        lines = f.readlines()
        
    layer_min_max={}
    name_order = []
    for idx, l in enumerate(lines):
        l = l.strip()
        if not l.startswith("min"):
            continue
        pre_l = lines[idx-1].strip()
        layer_name = pre_l.split(":")[1]
        if layer_name not in layer_min_max:
            name_order.append(layer_name)
            layer_min_max[layer_name] = {}
            layer_min_max[layer_name]["min"] = []
            layer_min_max[layer_name]["max99"] = []
            layer_min_max[layer_name]["max"] = []
        eles = l.split(",")
        layer_min_max[layer_name]["min"].append(float(eles[0].split(":")[1]))
        layer_min_max[layer_name]["max99"].append(float(eles[1].split(":")[1]))
        layer_min_max[layer_name]["max"].append(float(eles[2].split(":")[1]))
        
    for l in name_order:
        print("layer name: %s"% l)
        print("min:  min_%f, max_%f, mean_%f"%(  
              min(layer_min_max[l]["min"]),
              max(layer_min_max[l]["min"]),
              sum(layer_min_max[l]["min"])/len(layer_min_max[l]["min"])))
        print("max99:  min_%f, max_%f, mean_%f"%(  
              min(layer_min_max[l]["max99"]),
              max(layer_min_max[l]["max99"]),
              sum(layer_min_max[l]["max99"])/len(layer_min_max[l]["max99"])))
        print("max:  min_%f, max_%f, mean_%f"%(  
              min(layer_min_max[l]["max"]),
              max(layer_min_max[l]["max"]),
              sum(layer_min_max[l]["max"])/len(layer_min_max[l]["max"])))

if __name__ == "__main__":
    main()
