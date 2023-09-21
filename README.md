# Dataset
The [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) dataset was used to train the skin segmentation model
Steps to prepare the dataset:
1. Download the `.zip` file from the [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) repo (or use this link directly: [link](https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view?pli=1)
2. Place the `CelebAMask-HQ.zip` file in the projects root directory
3. Run data preparation script:
`python src/bin/prepare_CelebAMask_data.py`

# Skin segmentation model
The []() model was used for skin segmentation task
Run the following line to train the model (assuming that dataset is already prepared)
`python src/bin/train_segmentation_model.py`

# rPPG signal extraction
1. Segment all video frames to find the skin pixels for each frame
2. Average skin pixels for R, G and B channels

# Pulse estimation
1. Extract rPPG from video
2. Run FFT for RGB channels of rPPG signals to find the main frequency which is the Heart Rate (HR) 
