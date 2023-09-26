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
2. Filter out pixels belonging to the skin
3. Apply [Plane Orthogonal to the Skin (POS)](https://pure.tue.nl/ws/files/31563684/TBME_00467_2016_R1_preprint.pdf) algorithm to get POS signal from the RGB signals
4. Run FFT for RGB channels of rPPG signals to find the strongest frequency (which corresponds to the Heart Rate)
5. Calculate the Heart Rate using the strongest frequency: `HR = strongest_freq * 60`

# **Results**

## Processing pipeline
1. Raw frame
2. Frame trasformed to match model input size
3. Model output segmentation map (with all classes)
4. Output segmentation map filtered to match `skin` and `nose` labels (`skin_mask`)
5. Raw frame with pulse extracted for each pixel and masked with `skin_mask`

https://github.com/thawro/pulse-from-face-pytorch/assets/50373360/f953ff21-2666-4818-83f1-a0195a4d8905


## RGB and POS signals along with the frequency spectrum and estimated Heart Rate (HR). 
![signals_from_skin](https://github.com/thawro/pulse-from-face-pytorch/assets/50373360/d20750a0-e228-466c-b4ab-25f7ca2188a9)

