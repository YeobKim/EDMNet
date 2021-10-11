# EDMNet
Deblurring Network Using Edge Module, Deformable Convolution-Channel Attention Blocks and Multi-Stage Network

This is a PyTorch implementation of the my master's graduation paper in Hanyang University.


```
“This project is licensed under the terms of the MIT license.”
```

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

<p align="center"><img src="https://user-images.githubusercontent.com/59470033/136792497-03fc5c78-64d7-41bf-9442-06556c58d010.PNG" width="50%" height="50%"></p>


## YOLO v3 Results
| Blur | Deblur |
|---|---|
| ![yolo1](https://user-images.githubusercontent.com/59470033/136799347-bd1a2333-bc2e-4779-8d67-10257f94faf7.jpg) | ![yolo2](https://user-images.githubusercontent.com/59470033/136799359-87005673-5746-4622-ba0d-0fef079967f4.jpg) |
| ![yolo3](https://user-images.githubusercontent.com/59470033/136799492-853f535e-7248-4527-9791-a56f1b87351d.jpg) | ![yolo4](https://user-images.githubusercontent.com/59470033/136799524-2121c31b-e9b9-4b97-add9-e5de4c3359c0.jpg) |


## Ablation Study

## Contact
If you have any questions, please contact athurk94111@gmail.com.
