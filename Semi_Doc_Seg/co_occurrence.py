import json
from pathlib import Path
import numpy as np
import pandas as pd

with open("/home/abanerjee/Downloads/test.json", 'r') as f_json:
    instances = json.load(f_json)

print(list(instances.keys()))

cat_objs = instances["categories"]

cat_objs.sort(key=lambda x: x["id"])
num_classes = cat_objs[-1]["id"]

class_names = ["" for x in range(num_classes + 1)]

for x in cat_objs:
    class_names[x["id"] - 1] = x["name"]

img_to_class = {}

for x in instances["annotations"]:
    if x["image_id"] not in img_to_class.keys():
        img_to_class[x["image_id"]] = []
    img_to_class[x["image_id"]].append(x["category_id"])

#getting the counts of the occurence
counts = np.zeros((len(class_names), len(class_names)), dtype=int)

for k, v in img_to_class.items():
    v = list(set(v))
    v.sort()
    for ii in range(len(v)):
        for jj in range(ii, len(v)):
            counts[v[ii] - 1, v[jj] - 1] += 1

n_images = len(instances["images"])
diagonal_ind = np.diag(np.ones(len(class_names), dtype=bool))
class_counts = counts[diagonal_ind]
counts[diagonal_ind] = 0

counts += counts.T
counts[diagonal_ind] += class_counts

#calculating the probabilitites
class_probs = class_counts / n_images
jprobs = counts.astype(float) / n_images
cprobs = jprobs / class_probs[:, None]

#Creating the dataframe of the occurence matrix
class_probs = pd.DataFrame(class_probs, index=class_names)
counts = pd.DataFrame(counts, index=class_names, columns=class_names)
jprobs = pd.DataFrame(jprobs, index=class_names, columns=class_names)
cprobs = pd.DataFrame(cprobs, index=class_names, columns=class_names)

#dumping the results in csv file
counts.to_csv("counts.csv")
jprobs.to_csv("jprobs.csv")
cprobs.to_csv("cprobs.csv")
#class_probs.to_csv("class_probs.csv")
