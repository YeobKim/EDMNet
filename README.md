# EDMNet
Deblurring Network Using Edge Module, Deformable Convolution-Channel Attention Blocks and Multi-Stage Network

This is a PyTorch implementation of the my master's graduation paper. It is not yet complete. I will continue to update.

## Abstract
In this paper, we propse Deblurring Network Using Edge Module, Deformable Convolution-Channel Attention Blocks and Multi-Stage Network. The proposed network efficiently restores blurring objects using edge and deformable convolution. In addition, a total of three stages of learning perform using the multi-stage network. The sub-network of the multi-stage network consists of a U-net using the proposed DC-CAB. Also, we proposed RDFB(Residual Dense Feature Block) that learns about the original resolution not devided. The proposed network gets 31.52 dB of PSNR from the gopro test set and uses YOLO v3 to confirm object detection accuracy improvement. In addition, we perform an ablation study by removing the Edge Module and RDFB.

## Proposed Network
- Network Architecture
![net](https://user-images.githubusercontent.com/59470033/136790773-733ab1ba-ce79-4b87-8227-75d4501b0b68.png)

## Experimental Results

