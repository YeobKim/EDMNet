# EDMNet
Deblurring Network Using Edge Module, Deformable Convolution-Channel Attention Blocks and Multi-Stage Network

This is a PyTorch implementation of the my master's graduation paper. It is not yet complete. I will continue to update.

## Abstract
In this paper, we propse Deblurring Network Using Edge Module, Deformable Convolution-Channel Attention Blocks and Multi-Stage Network. The proposed network efficiently restores blurring objects using edge and deformable convolution. In addition, a total of three stages of learning perform using the multi-stage network. The sub-network of the multi-stage network consists of a U-net using the proposed DC-CAB. Also, we proposed RDFB(Residual Dense Feature Block) that learns about the original resolution not devided. The proposed network gets 31.52 dB of PSNR from the gopro test set and uses YOLO v3 to confirm object detection accuracy improvement. In addition, we perform an ablation study by removing the Edge Module and RDFB.

## Proposed Network
- Network Architecture
![EDMNet_rfa](https://user-images.githubusercontent.com/59470033/136790972-9984d09f-f93d-41ee-9edd-bd1f143ba685.png)

- Sub-Network
<p align="center"><img src="https://user-images.githubusercontent.com/59470033/136791164-efc7f952-fd81-41ac-8075-2d1857b9ddef.png" width="70%" height="70%"></p>

- Deformable Convolution Attention Block
<p align="center"><img src="https://user-images.githubusercontent.com/59470033/136791190-9cb25267-910f-4896-8371-89ea9eeeb9fc.png" width="70%" height="70%"></p>


## Run
* train
```
python train.py
```

* test
```
python test.py
```

## Dataset
We used the [GoPro](https://seungjunnah.github.io/Datasets/gopro) datasets for training and testing the proposed network.   
There are other options you can choose. Please refer to dataset.py.

## Experimental Results
- PSNR & SSIM
![result](https://user-images.githubusercontent.com/59470033/136792497-03fc5c78-64d7-41bf-9442-06556c58d010.PNG)


## Ablation Study

## Contact
If you have any questions, please contact athurk94111@gmail.com.
