# SemiDocSeg

## Description
Pytorch implementation of the paper [SemiDocSeg: Harnessing Semi-Supervised Learning
for Document Layout Analysis]([https://arxiv.org/abs/2305.04609](https://assets.researchsquare.com/files/rs-3611689/v1_covered_95734a88-9a07-4fdb-ae11-1aac4b5410fe.pdf?c=1700210451)).
This model is also implemented on top of the [detectron2](https://github.com/facebookresearch/detectron2) framework.

## Generating Support Set
```bash
cd Semi_Doc_seg/datasets
sh generate_support_data.sh
```

Hurray!!! Now take your new_annotations and train the SwinDocSegmenter!!!.
