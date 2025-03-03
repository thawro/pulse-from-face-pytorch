import cv2
import time
import torch
import numpy as np
from functools import partial

from src.model.model.segmentation import SegmentationModel
from src.model.load import load_model
from src.data.transforms import CelebATransform
from src.visualization.segmentation import colorize_mask
from src.bin.segmentation.config import (
    IMGSZ,
    MEAN,
    STD,
    N_CLASSES,
    DEVICE,
    MODEL_INPUT_SIZE,
    PALETTE,
    LABELS,
)
from src.bin.utils import POSExtractor, plot_signals, filter_pos
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

transform = CelebATransform(IMGSZ, MEAN, STD)

CKPT_PATH = str(ROOT / "results/test/23-09-2023_07:54:35/checkpoints/last.pt")
FPS = 30
POS_EXTRACTOR = POSExtractor(FPS).to(DEVICE)

MODE = "segmentation"
EVAL_RESULTS_PATH = ROOT / "evaluation" / MODE
EVAL_RESULTS_PATH.mkdir(exist_ok=True, parents=True)

VIDEO_IN_PATH = 0  # "evaluation/input/video_2.MOV"
VIDEO_IN_PATH = "evaluation/input/video_2.MOV"

VIDEO_OUT_PATH = str(EVAL_RESULTS_PATH / "pulse_extraction.mp4")
SIGNALS_OUT_PATH = str(EVAL_RESULTS_PATH / "signals.jpg")


def predict(image: np.ndarray, model: SegmentationModel, labels: list[str]):
    input_image, resize_size, crop_coords = transform.process(image)
    seg_mask = model.segment(input_image.unsqueeze(0).to(DEVICE))
    input_image = transform.inverse_preprocessing(input_image)
    seg_mask = seg_mask.squeeze().argmax(0)
    output_mask = filter_labels_from_mask(seg_mask, labels_to_filer=labels, all_labels=LABELS)
    output_mask = output_mask.cpu().numpy()
    seg_mask = seg_mask.cpu().numpy()
    output_mask = keep_largest_blob(output_mask)
    # kernel = np.ones((11, 11), np.uint8)
    # output_mask = cv2.morphologyEx(output_mask, cv2.MORPH_CLOSE, kernel)
    return seg_mask, output_mask, input_image, resize_size, crop_coords


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


def extract_rgb_from_segmentation(
    frame: np.ndarray,
    model: SegmentationModel,
    prev_frames: torch.Tensor,
    idx: int,
    labels: list[str],
):
    start_time = time.time()
    frame = avg_pool(frame, kernel=(4, 4))
    frame_h, frame_w = frame.shape[:2]

    seg_mask, output_mask, input_frame, resize_size, crop_coords = predict(frame, model, labels)
    frame_mask = inverse_processing_mask(output_mask, crop_coords, resize_size, (frame_h, frame_w))

    # TODO
    contours, _ = cv2.findContours(
        (frame_mask * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    max_contour = max(contours, key=cv2.contourArea)
    cv2.imshow("Frame mask", (frame_mask * 255).astype(np.uint8))
    # rect = cv2.minAreaRect(contours[0])
    # face_rect_coords = cv2.boxPoints(rect).astype(np.int32)
    x1, y1, w, h = cv2.boundingRect(max_contour)
    x2 = x1 + w
    y2 = y1 + h
    face_rect_coords = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    dst_h, dst_w = 500, 500
    pts1 = np.float32(face_rect_coords)
    pts2 = np.float32([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    face_rect = cv2.warpPerspective(frame.astype(np.uint8), M, (dst_w, dst_h))
    cv2.imshow("Face rect", cv2.cvtColor(face_rect, cv2.COLOR_RGB2BGR))
    # TODO

    skin_mask = frame_mask == 1
    face_rgb = frame[skin_mask].mean(0)  # H x W x 3 -> 3
    RGB_SIGNAL[idx] = torch.from_numpy(face_rgb)

    prev_frames = prev_frames.roll(shifts=-1, dims=0)
    prev_frames[-1] = convolve_gaussian_2d(frame, 27, 3, DEVICE)

    win_len = POS_EXTRACTOR.win_len
    start_idx = idx - win_len
    if start_idx >= 0 and start_idx < len(POS_SIGNAL) - win_len:
        end_idx = start_idx + win_len
        face_rgb_input = RGB_SIGNAL[start_idx:end_idx].unsqueeze(0).to(DEVICE)
        pos_window = POS_EXTRACTOR(face_rgb_input).cpu().squeeze()
        POS_SIGNAL[start_idx:end_idx] += pos_window
        pos = POS_SIGNAL[:end_idx].cpu().numpy()

        pos = filter_pos(pos, FPS).copy()
        pos = minmax(pos, dim=0, keepdim=True, scaler=1).numpy()
        scaler = pos[start_idx]  # get value at start_idx since it is the proper POS approx
        pt1 = (start_idx - 1, int(pos[start_idx - 1] * 255) + 1)
        pt2 = (start_idx, int(pos[start_idx] * 255) + 1)
        cv2.line(POS_CANVAS, pt1, pt2, GREEN, 1)
    else:
        scaler = 1
    prev_skin_pixels = prev_frames[:, skin_mask]
    prev_skin_pixels = minmax(prev_skin_pixels, dim=0, keepdim=True, scaler=255)
    last_frame_pixels = prev_skin_pixels[0].mean(-1)  # [..., 1] for G ,  .mean(-1) for RGB
    last_frame_pixels = (last_frame_pixels * scaler).cpu().numpy().astype(np.uint8)
    last_frame_pixels = cv2.applyColorMap(last_frame_pixels, cv2.COLORMAP_JET).squeeze()

    pulse_mask = np.zeros((*skin_mask.shape, 3), dtype=np.uint8)
    pulse_mask[skin_mask] = last_frame_pixels

    frame = frame.astype(np.uint8)
    face_pulse = cv2.addWeighted(frame, 1, pulse_mask, 0.4, 1)
    seg_mask = colorize_mask(seg_mask, PALETTE)

    output_mask = cv2.normalize(output_mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    video_frames = [frame, input_frame, seg_mask, output_mask, face_pulse]
    labels = ["Raw frame", "Model input", "Model output", "Skin", "Face pulse"]
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

    process_fn = partial(extract_rgb_from_segmentation, model=model, labels=["skin", "nose"])

    result = process_video(process_fn, prev_frames, VIDEO_IN_PATH, start_frame=0, end_frame=-1)
    save_frames_to_video(result["pulse_extraction"], FPS, VIDEO_OUT_PATH)

    rgb = np.stack(result["rgb"])
    rgb_input = torch.from_numpy(rgb).unsqueeze(0).to(DEVICE).float()
    pos = POS_EXTRACTOR(rgb_input).squeeze().cpu().numpy()

    plot_signals(rgb, pos, FPS, filepath=SIGNALS_OUT_PATH)


if __name__ == "__main__":
    main()
