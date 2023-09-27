import cv2
import time
import torch
import numpy as np
from functools import partial

from src.model.model.segmentation import SegmentationModel
from src.model.load import load_model
from src.data.transforms import CelebATransform
from src.visualization.segmentation import colorize_mask
from src.bin.config import IMGSZ, MEAN, STD, N_CLASSES, DEVICE, MODEL_INPUT_SIZE, PALETTE, LABELS
from src.bin.eval.utils import POSExtractor, plot_signals
from src.utils.config import ROOT
from src.utils.video import process_video, save_frames_to_video, get_video_params
from src.utils.ops import (
    keep_largest_blob,
    avg_pool,
    convolve_gaussian_2d,
    minmax,
    filter_labels_from_mask,
)
from src.utils.image import (
    stack_frames_horizontally,
    stack_frames_vertically,
    add_txt_to_image,
    add_labels_to_frames,
    GREEN,
)

transform = CelebATransform(IMGSZ, MEAN, STD)  # TODO: change

CKPT_PATH = str(ROOT / "results/test/23-09-2023_07:54:35/checkpoints/last.pt")  # TODO: change
FPS = 30
POS_EXTRACTOR = POSExtractor(FPS).to(DEVICE)
MODE = "keypoints"
EVAL_RESULTS_PATH = ROOT / "evaluation" / MODE
EVAL_RESULTS_PATH.mkdir(exist_ok=True, parents=True)

VIDEO_IN_PATH = "evaluation/input/video_2.MOV"
VIDEO_OUT_PATH = str(EVAL_RESULTS_PATH / "pulse_extraction.mp4")
SIGNALS_OUT_PATH = str(EVAL_RESULTS_PATH / "signals.jpg")


def predict(image: np.ndarray, model):
    input_image, resize_size, crop_coords = transform.process(image)
    out = model(input_image.unsqueeze(0).to(DEVICE))
    input_image = transform.inverse_preprocessing(input_image)
    return out, input_image, resize_size, crop_coords


def inverse_processing_mask(
    output_mask: np.ndarray,
    crop_coords: tuple[int, int, int, int],
    resize_size: tuple[int, int],
    frame_size: tuple[int, int],
):
    resize_w, resize_h = resize_size
    frame_h, frame_w = frame_size
    xmin, ymin, xmax, ymax = crop_coords
    frame_mask = np.zeros((resize_h, resize_w))
    frame_mask[ymin:ymax, xmin:xmax] = output_mask
    frame_mask = cv2.resize(frame_mask, (frame_w, frame_h))
    return frame_mask


def extract_rgb_from_keypoints(
    frame: np.ndarray, model: SegmentationModel, prev_frames: torch.Tensor, idx: int
):
    start_time = time.time()
    frame = avg_pool(frame, kernel=(4, 4))
    frame_h, frame_w = frame.shape[:2]

    out, input_frame, resize_size, crop_coords = predict(frame, model)
    face_rgb = frame.mean((0, 1))  # H x W x 3 -> 3

    prev_frames = prev_frames.roll(shifts=-1, dims=0)
    prev_frames[-1] = convolve_gaussian_2d(frame, 27, 3, DEVICE)

    frame = frame.astype(np.uint8).copy()

    keypoints = np.zeros_like(input_frame, dtype=np.uint8)
    frame_keypoints = np.zeros_like(frame, dtype=np.uint8)
    face_pulse = np.zeros_like(frame, dtype=np.uint8)

    video_frames = [frame, input_frame, keypoints, frame_keypoints, face_pulse]
    labels = ["Raw frame", "Model input", "Model output", "Frame with keypoints", "Face pulse"]
    video_frames = add_labels_to_frames(video_frames, labels)
    pulse_extraction = stack_frames_horizontally(video_frames)
    add_txt_to_image(POS_CANVAS, ["POS approximation"], loc="tc")
    pulse_extraction = stack_frames_vertically([pulse_extraction, POS_CANVAS])

    duration_ms = (time.time() - start_time) * 1000
    add_txt_to_image(pulse_extraction, [f"FPS: {int(1000 / duration_ms)}"], loc="tc")

    cv2.imshow("Raw frame", cv2.cvtColor(pulse_extraction, cv2.COLOR_RGB2BGR))
    return {"rgb": face_rgb, "pulse_extraction": pulse_extraction}, prev_frames


def main():
    model = load_model(N_CLASSES, MODEL_INPUT_SIZE, CKPT_PATH, DEVICE)

    video_params = get_video_params(VIDEO_IN_PATH)
    video_h = video_params["height"]
    video_w = video_params["width"]
    video_frame_count = int(video_params["frame_count"])
    h, w = int(video_h // 4), int(video_w // 4)

    if video_frame_count < 0:
        video_frame_count = 1000

    global POS_CANVAS, POS_SIGNAL, RGB_SIGNAL
    POS_CANVAS = np.zeros((256 + 2, video_frame_count, 3))
    POS_SIGNAL = torch.zeros(video_frame_count)
    RGB_SIGNAL = torch.zeros(video_frame_count, 3)
    prev_frames = (torch.zeros(POS_EXTRACTOR.win_len, h, w, 3)).to(DEVICE)

    process_fn = partial(extract_rgb_from_keypoints, model=model)
    result = process_video(process_fn, prev_frames, VIDEO_IN_PATH, start_frame=0, end_frame=-1)

    save_frames_to_video(result["pulse_extraction"], FPS, VIDEO_OUT_PATH)
    rgb = np.stack(result["rgb"])
    rgb_input = torch.from_numpy(rgb).unsqueeze(0).float()
    pos = POS_EXTRACTOR(rgb_input.to(DEVICE)).squeeze().cpu().numpy()

    plot_signals(rgb, pos, FPS, filepath=SIGNALS_OUT_PATH)


if __name__ == "__main__":
    main()
