# Dataset
The [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) dataset was used to train the skin segmentation model
Steps to prepare the dataset:
1. Download the `.zip` file from the [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) repo (or use this link directly: [link](https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view?pli=1)
2. Place the `CelebAMask-HQ.zip` file in the projects root directory
3. Run data preparation script:
`python src/bin/prepare_data.py`

# Skin segmentation model
The [PSPNet](https://arxiv.org/abs/1612.01105) model was used for skin segmentation task (trained from scratch on CeleAMask-HQ dataset).
Run the following line to train the model (assuming that dataset is already prepared)
`python src/bin/celebA/train.py`

# rPPG signal extraction
To extract rPPG signal from the face:
1. Segment all video frames to find the skin pixels for each frame
2. Apply erosion to make the skin mask a bit smaller to prevent mask leak out of the face
3. Average skin pixels for R, G and B channels


# Pulse estimation
1. Extract rPPG from video - after this step the R, G and B signals are present (signal length is equal to number of frames)
2. Apply [Plane Orthogonal to the Skin (POS)](https://pure.tue.nl/ws/files/31563684/TBME_00467_2016_R1_preprint.pdf) algorithm to get POS signal from the RGB signals
3. Run FFT for RGB channels of rPPG signals to find the strongest frequency (which corresponds to the Heart Rate)
4. Calculate the Heart Rate using the strongest frequency: `HR = strongest_freq * 60`

# **Results**

## The input video with skin mask predicted by the `PSPNet` model:
https://github.com/thawro/pulse-from-face-pytorch/assets/50373360/e8312e1e-00b7-4867-82b3-01f2ea53f89b

## RGB and POS signals along with the frequency spectrum and estimated Heart Rate (HR). 
![signals](https://github.com/thawro/pulse-from-face-pytorch/assets/50373360/7a51c9ae-068f-4c4b-9a7e-c6cd4007e6ad)

