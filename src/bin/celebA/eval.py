from src.utils.config import ROOT
from src.model.model.segmentation import SegmentationModel
from src.model.architectures.psp_net import PSPNet
from src.data.transforms import CelebATransform
import numpy as np
import torch
from src.visualization.segmentation import colorize_mask
from src.utils.ops import keep_largest_blob
import cv2
from scipy import signal

from src.bin.celebA.config import (
    IMGSZ,
    MEAN,
    STD,
    NUM_CLASSES,
    DEVICE,
    BATCHED_MODEL_INPUT_SIZE,
    PALETTE,
    LABELS,
)
from src.utils.utils import find_center_of_mass
import time
from src.utils.video import record_webcam_to_mp4, process_video, save_frames_to_video
from functools import partial
import matplotlib.pyplot as plt

CKPT_PATH = ROOT / "results/test/23-09-2023_07:54:35/checkpoints/last.pt"
transform = CelebATransform(IMGSZ, MEAN, STD)

RED = (0, 0, 255)
SIZE = IMGSZ // 6
HALF_SIZE = int(SIZE // 2)


def load_model():
    model = SegmentationModel(
        net=PSPNet(num_classes=NUM_CLASSES, cls_dropout=0.5, backbone="resnet101"),
        input_size=BATCHED_MODEL_INPUT_SIZE,
        input_names=["images"],
        output_names=["masks", "class_probs"],
    )
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    ckpt = ckpt["module"]["model"]
    model.load_state_dict(ckpt)
    return model.to(DEVICE)


def extract_labels(mask: torch.Tensor, labels: str | list[str]) -> torch.Tensor:
    new_mask = torch.zeros_like(mask, dtype=torch.float32)
    if isinstance(labels, str):
        labels = [labels]
    for label in labels:
        idx = LABELS.index(label) + 1  # +1 for background
        new_mask[mask == idx] = 1
    return new_mask


def minmax(lst):
    arr = np.array(lst)
    return (arr - arr.min()) / (arr.max() - arr.min())


def plot_timeseries(arr):
    H = 400
    W = 500

    margin = H // 8
    arr = minmax(arr)
    arr = arr * (H - margin * 2) + margin
    arr = arr.astype(np.int32).tolist()

    base = np.ones((H, W, 3)) * 255
    cv2.line(base, (0, H // 2), (W, H // 2), (0, 255, 0), 1)
    for i in range(len(arr) - 1):
        pt1 = (i, arr[i])
        pt2 = (i + 1, arr[i + 1])
        cv2.line(base, pt1, pt2, (255, 0, 0), 1)
    cv2.imshow("rPPG", base)


class POSExtractor(torch.nn.Module):
    def __init__(self, fps: float):
        super().__init__()
        self.fs = fps
        self.win_sec = 1.6

    def forward(self, x):
        batch_size, N, num_features = x.shape
        H = torch.zeros(batch_size, 1, N).to("cuda")

        for b in range(batch_size):
            RGB = x[b]  # Assume RGB preprocessing already done
            N = RGB.shape[0]
            l = int(self.fs * self.win_sec)  # math.ceil(WinSec * fs)

            for n in range(N):
                m = n - l
                if m >= 0:
                    Cn = RGB[m:n, :] / torch.mean(RGB[m:n, :], dim=0)
                    Cn = torch.transpose(Cn, 0, 1)
                    S = torch.matmul(
                        torch.tensor([[0, 1, -1], [-2, 1, 1]], dtype=torch.float).to("cuda"), Cn
                    )
                    h = S[0, :] + (torch.std(S[0, :]) / torch.std(S[1, :])) * S[1, :]
                    mean_h = torch.mean(h)
                    h = h - mean_h
                    H[b, 0, m:n] = H[b, 0, m:n] + h
        return H


def predict(frame: np.ndarray, model: SegmentationModel, labels: list[str] | None = None):
    _frame, resize_size, crop_coords = transform.process(frame)
    seg_mask, cls_pred = model(_frame.unsqueeze(0).to(DEVICE))
    _frame = transform.inverse_preprocessing(_frame)
    seg_mask = seg_mask.squeeze().argmax(0)
    if labels is not None:
        seg_mask = extract_labels(seg_mask, labels=labels)
    return seg_mask.cpu().numpy(), _frame, resize_size, crop_coords


def extract_rgb_from_forehead(frame: np.ndarray, model: SegmentationModel):
    frame_h, frame_w = frame.shape[:2]

    mask, _frame, resize_size, crop_coords = predict(frame, model, ["nose"])
    mask_h, mask_w = mask.shape

    h_ratio = frame_h / mask_h
    w_ratio = frame_w / mask_w

    yc, xc = find_center_of_mass(mask)  # nose
    yc -= 70  # middle of forehead

    yc, xc = int(yc * h_ratio), int(xc * w_ratio)
    size = 60

    xmin, ymin = xc - size, yc - size
    xmax, ymax = xc + size, yc + size

    box = frame[ymin:ymax, xmin:xmax]  # forehead
    rgb = box.mean(axis=(0, 1))  # H x W x 3 -> 3

    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), RED, 1)
    cv2.imshow("Raw frame", frame)
    cv2.imshow("Model input", _frame)
    cv2.imshow("Mask", mask)

    return {"rgb": rgb}


def extract_rgb_from_skin(frame: np.ndarray, model: SegmentationModel):
    frame_h, frame_w = frame.shape[:2]
    mask, _frame, resize_size, crop_coords = predict(frame, model, ["skin"])
    xmin, ymin, xmax, ymax = crop_coords
    resize_w, resize_h = resize_size

    mask_before_crop = np.zeros((resize_h, resize_w))

    mask = keep_largest_blob(mask)
    mask = cv2.erode(mask, kernel=np.ones((5, 5), np.uint8), iterations=4)

    mask_before_crop[ymin:ymax, xmin:xmax] = mask
    mask = cv2.resize(mask_before_crop, (frame_w, frame_h))

    bool_mask = mask == 1

    rgb = frame[bool_mask].mean(0)  # H x W x 3 -> 3

    mask = (mask * 255).astype(np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    frame_with_mask = cv2.addWeighted(frame, 1, mask, 0.3, 1)

    cv2.imshow("Raw frame", cv2.cvtColor(frame_with_mask, cv2.COLOR_RGB2BGR))
    cv2.imshow("Model input", cv2.cvtColor(_frame, cv2.COLOR_RGB2BGR))

    return {"rgb": rgb, "frame_mask": frame_with_mask}


def get_signal_frequencies(signal: np.ndarray, fs: float):
    y_fft = np.fft.fft(signal)  # Original FFT
    y_fft = y_fft[: round(len(signal) / 2)]  # First half ( pos freqs )
    y_fft = np.abs(y_fft)  # Absolute value of magnitudes
    y_fft = y_fft / max(y_fft)  # Normalized so max = 1
    freq_x_axis = np.linspace(0, fs / 2, len(y_fft))
    return freq_x_axis, y_fft


def plot_signals(
    rgb: np.ndarray, pos: np.ndarray, freqs: np.ndarray, pos_fft: np.ndarray, fps: float
):
    duration = len(rgb) / fps
    t = np.arange(0, duration, 1 / fps)
    fig, axes = plt.subplots(3, 1, figsize=(14, 15))
    colors = ["r", "g", "b"]
    for i, sig in enumerate(rgb.T):
        axes[0].plot(t, sig, c=colors[i], label=colors[i].capitalize())
    axes[0].legend()
    axes[0].set_title("RGB")

    strongest_freq = freqs[np.argmax(pos_fft)]
    heart_rate = 60 * strongest_freq  # beats per minute [BPM]
    axes[1].plot(t, pos)
    axes[1].set_title(f"POS signal, Heart Rate: {heart_rate:.2f} bpm")

    axes[2].plot(freqs, pos_fft, "o-")
    axes[2].set_title("Frequency magnitudes")

    fig.savefig("temp/signals.jpg", bbox_inches="tight")


def main():
    filename = "temp/video.mp4"
    filename = "temp/video_2.MOV"

    # record_webcam_to_mp4(filename)
    # exit()
    model = load_model()
    processing_fn = partial(extract_rgb_from_skin, model=model)
    # processing_fn = partial(extract_rgb_from_forehead, model=model)

    # filename = 0  # webcam
    result = process_video(processing_fn, filename=filename, start_frame=0, end_frame=-1)
    rgb = np.stack(result["rgb"])
    frames = result["frame_mask"]
    save_frames_to_video(frames, 30, "temp/frames_mask.mp4")
    np.save("temp/rgb.npy", rgb)
    rgb = np.load("temp/rgb.npy")

    fps = 30
    pos_extractor = POSExtractor(fps).to(DEVICE)

    rgb_input = torch.from_numpy(rgb).unsqueeze(0).to(DEVICE).float()
    pos = pos_extractor(rgb_input).squeeze().cpu().numpy()

    b, a = signal.butter(1, [0.75 / fps * 2, 3 / fps * 2], btype="bandpass")
    pos = signal.filtfilt(b, a, pos.astype(np.double))

    freqs, pos_fft = get_signal_frequencies(pos, fps)
    plot_signals(rgb, pos, freqs, pos_fft, fps)


if __name__ == "__main__":
    main()
