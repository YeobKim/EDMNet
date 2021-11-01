# EDMNet
Deblurring Network Using Edge Module, Deformable Convolution-Channel Attention Blocks and Multi-Stage Network

This is a PyTorch implementation of the my master's graduation paper in Hanyang University.


```
“This project is licensed under the terms of the MIT license.”
```

## Abstract
In this paper, we propse Deblurring Network Using Edge Module, Deformable Convolution-Channel Attention Blocks and Multi-Stage Network. The proposed network efficiently restores blurring objects using edge and deformable convolution. In addition, a total of three stages of learning perform using the multi-stage network. The sub-network of the multi-stage network consists of a U-net using the proposed DC-CAB. Also, we proposed RDFB(Residual Dense Feature Block) that learns about the original resolution not devided. The proposed network gets 31.88 dB of PSNR from the gopro test set and uses YOLO v3 to confirm object detection accuracy improvement. In addition, we perform an ablation study by removing the Edge Module and RDFB.

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
### PSNR & SSIM

<p align="center"><img src="https://user-images.githubusercontent.com/59470033/138881235-64c49389-6a30-4324-a222-6a472c001c61.PNG" width="70%" height="70%"></p>

### Qualitative comparison
| GroundTruth | Blurry | DeepDeblur | SRN |
|---|---|---|---|
| ![gt](https://user-images.githubusercontent.com/59470033/139620953-d9bb9ff6-d871-4006-ab13-fe5d690709bf.png)|![blurry](https://user-images.githubusercontent.com/59470033/139621006-0c783ed4-3730-4cfb-bc96-6f07ec512aa6.png)|![DeepDeblur](https://user-images.githubusercontent.com/59470033/139621138-7acc40e9-e9c7-435d-a927-ec4f245a3e35.png)|![SRN](https://user-images.githubusercontent.com/59470033/139621156-70b3056d-4e73-4bf5-b2ab-81bbdce4e946.png)|

| Gao et al. | DBGAN | DMPHN | EDMNet(Ours) |
|![Gao](https://user-images.githubusercontent.com/59470033/139621231-cce04419-cc38-4131-bda9-a545683a92a1.png)|![DBGAN](https://user-images.githubusercontent.com/59470033/139621367-2a430064-63c5-4659-afe1-d69ac3a1353a.png)|![DMPHN](https://user-images.githubusercontent.com/59470033/139621380-a313472f-0718-4fbe-8111-43a2a09c1114.png)|![Proposed](https://user-images.githubusercontent.com/59470033/139621389-0e5a9c87-313b-4c9f-b559-0be19c9f9437.png)|

### YOLO v3 Results
| Blur | Deblur |
|---|---|
| ![yolo1](https://user-images.githubusercontent.com/59470033/136799347-bd1a2333-bc2e-4779-8d67-10257f94faf7.jpg) | ![yolo2](https://user-images.githubusercontent.com/59470033/136799359-87005673-5746-4622-ba0d-0fef079967f4.jpg) |
| ![yolo3](https://user-images.githubusercontent.com/59470033/136799492-853f535e-7248-4527-9791-a56f1b87351d.jpg) | ![yolo4](https://user-images.githubusercontent.com/59470033/136799524-2121c31b-e9b9-4b97-add9-e5de4c3359c0.jpg) |


## Ablation Study
<p align="center"><img src="https://user-images.githubusercontent.com/59470033/139620539-a637eb29-d22d-4790-b796-41751de26268.png" width="70%" height="70%"></p>

## Contact
If you have any questions, please contact athurk94111@gmail.com.
