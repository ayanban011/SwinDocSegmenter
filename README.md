# SwinDocSegmenter

## Description
Pytorch implementation of the paper [SwinDocSegmenter: An End-to-End Unified Domain Adaptive Transformer for Document Instance Segmentation](https://arxiv.org/abs/2305.04609). This model is implemented on top of the [detectron2](https://github.com/facebookresearch/detectron2) framework. The proposed model can be used to analysis the complex layouts including [magazines](https://www.primaresearch.org/datasets/Layout_Analysis), [Scientific Reports](https://github.com/ibm-aur-nlp/PubLayNet), [historical documents](https://dell-research-harvard.github.io/HJDataset/), [patents](https://github.com/DS4SD/DocLayNet) and so on as shown in the following examples.

<table style="padding:10px">
    <tr>
        <td style="text-align:center">
            Magazines 
        </td>
        <td style="text-align:center">
            Scientific Reports> 
        </td>
    </tr>
    <tr>
        <td style="text-align:center"> 
            <img src="./git_images/3.png"  alt="1" width = 600px height = 300px >
        </td>
        <td style="text-align:center">
            <img src="./git_images/3_pred.png"  alt="2" width = 600px height = 300px>
        </td>
    </tr>
    <tr>
        <td style="text-align:center">
            Historical Document 
        </td>
        <td style="text-align:center">
            Others 
        </td>
    </tr>
    <tr>
        <td style="text-align:center"> 
            <img src="./git_images/14.png"  alt="1" width = 600px height = 300px >
        </td>
        <td style="text-align:center">
            <img src="./git_images/14_pred.png"  alt="2" width = 600px height = 300px>
        </td>
    </tr>

</table>

