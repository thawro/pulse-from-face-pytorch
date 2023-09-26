import cv2
import time
import torch
import numpy as np
from scipy import signal
from functools import partial
import matplotlib.pyplot as plt

from src.model.model.segmentation import SegmentationModel
from src.model.architectures.psp_net import PSPNet
from src.data.transforms import CelebATransform
from src.visualization.segmentation import colorize_mask
from src.bin.config import IMGSZ, MEAN, STD, N_CLASSES, DEVICE, MODEL_INPUT_SIZE, PALETTE, LABELS
from src.utils.config import ROOT
from src.utils.video import process_video, save_frames_to_video, get_video_size
from src.utils.ops import (
    keep_largest_blob,
    avg_pool,
    convolve_gaussian_2d,
    minmax,
    filter_labels_from_mask,
    find_center_of_mass,
)
from src.utils.image import (
    stack_frames_horizontally,
    add_txt_to_image,
    add_labels_to_frames,
    RED,
    GREEN,
    BLUE,
)

transform = CelebATransform(IMGSZ, MEAN, STD)

CKPT_PATH = ROOT / "results/test/23-09-2023_07:54:35/checkpoints/last.pt"
FPS = 30


def load_model():
    model = SegmentationModel(
        net=PSPNet(num_classes=N_CLASSES, cls_dropout=0.5, backbone="resnet101"),
        input_size=MODEL_INPUT_SIZE,
        input_names=["images"],
        output_names=["masks", "class_probs"],
    )
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    ckpt = ckpt["module"]["model"]
    model.load_state_dict(ckpt)
    return model.to(DEVICE)


class POSExtractor(torch.nn.Module):
    def __init__(self, fps: float):
        super().__init__()
        win_len_sec = 1.6
        win_len = int(win_len_sec * fps)
        self.s_proj = torch.nn.Linear(3, 2)
        with torch.no_grad():
            M = torch.tensor([[0, 1, -1], [-2, 1, 1]], dtype=torch.float).to(DEVICE)
            self.s_proj.weight = torch.nn.Parameter(M)

        self.win_len = win_len

    def forward(self, rgbs):
        B, N, C = rgbs.shape
        all_rgbs = []
        for _rgb in rgbs:
            _rgb = _rgb.unfold(dimension=0, size=self.win_len, step=1)
            _rgb = _rgb.permute(0, 2, 1)
            all_rgbs.append(_rgb)
        all_rgbs = torch.stack(all_rgbs)
        rgb = all_rgbs.flatten(0, 1)
        fold_B, _, C = rgb.shape
        H = torch.zeros(B, N).to(DEVICE)
        Cn = rgb / rgb.mean(dim=1, keepdim=True)

        S = self.s_proj(Cn.flatten(0, 1)).reshape(fold_B, self.win_len, 2).permute(0, 2, 1)

        std_0 = S[:, 0].std(1)
        std_1 = S[:, 1].std(1)

        h = S[:, 0] + (std_0 / std_1).unsqueeze(1) * S[:, 1]
        h = h - h.mean(1, keepdim=True)
        i = 0
        for b in range(B):
            for m in range(0, N - self.win_len):
                n = m + self.win_len
                H[b, m:n] += h[i]
                i += 1
        return H.detach()


def predict(frame: np.ndarray, model: SegmentationModel, labels: list[str]):
    input_frame, resize_size, crop_coords = transform.process(frame)
    seg_mask, _ = model(input_frame.unsqueeze(0).to(DEVICE))
    input_frame = transform.inverse_preprocessing(input_frame)

    seg_mask = seg_mask.squeeze().argmax(0)
    output_mask = filter_labels_from_mask(seg_mask, labels_to_filer=labels, all_labels=LABELS)
    output_mask = output_mask.cpu().numpy()
    seg_mask = seg_mask.cpu().numpy()
    output_mask = keep_largest_blob(output_mask)
    return seg_mask, output_mask, input_frame, resize_size, crop_coords


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


def extract_rgb_from_box(
    frame: np.ndarray,
    model: SegmentationModel,
    box_size: tuple[int, int],
    prev_frames: torch.Tensor,
):
    start_time = time.time()
    frame = avg_pool(frame, kernel=(4, 4))
    frame_h, frame_w = frame.shape[:2]

    seg_mask, output_mask, input_frame, resize_size, crop_coords = predict(
        frame, model, ["skin", "nose"]
    )
    skin_frame_mask = inverse_processing_mask(
        output_mask, crop_coords, resize_size, (frame_h, frame_w)
    )
    skin_frame_mask = np.expand_dims(skin_frame_mask.astype(np.uint8), -1)

    mask_h, mask_w = output_mask.shape

    h_ratio = frame_h / mask_h
    w_ratio = frame_w / mask_w

    yc, xc = find_center_of_mass(output_mask)  # nose
    # yc, xc = 122, 128

    yc, xc = int(yc * h_ratio), int(xc * w_ratio)
    box_h, box_w = box_size

    xmin, ymin = xc - box_w // 2, yc - box_h // 2
    xmax, ymax = xc + box_w // 2, yc + box_h // 2

    box_mask = np.zeros(frame.shape[:2])
    box_mask[ymin:ymax, xmin:xmax] = 1
    box_mask = box_mask * skin_frame_mask.squeeze()
    skin_mask = box_mask == 1

    global_rgb = frame[skin_mask].mean(0)  # H x W x 3 -> 3

    prev_frames = prev_frames.roll(shifts=-1, dims=2)
    prev_frames[..., -1] = convolve_gaussian_2d(frame[..., 1], 27, 3, DEVICE)[ymin:ymax, xmin:xmax]

    prev_skin_pixels = minmax(prev_frames, dim=-1, keepdim=True, scaler=255)
    last_frame_pixels = prev_skin_pixels[..., -1]
    last_frame_pixels = last_frame_pixels.cpu().numpy().astype(np.uint8)
    last_frame_pixels = cv2.applyColorMap(last_frame_pixels, cv2.COLORMAP_JET)
    pulse_mask = np.zeros((*skin_mask.shape, 3), dtype=np.uint8)
    pulse_mask[ymin:ymax, xmin:xmax] = last_frame_pixels

    frame = frame.astype(np.uint8).copy()
    pulse_mask = pulse_mask * skin_frame_mask
    face_pulse = cv2.addWeighted(frame, 1, pulse_mask, 0.4, 1)

    seg_mask = colorize_mask(seg_mask, PALETTE)

    frame_roi = frame.copy()
    cv2.rectangle(frame_roi, (xmin, ymin), (xmax, ymax), GREEN, 2)
    cv2.circle(frame_roi, (xc, yc), 7, RED, -1)

    output_mask = cv2.normalize(output_mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    video_frames = [frame, input_frame, seg_mask, output_mask, frame_roi, face_pulse]
    labels = ["Raw frame", "Model input", "Model output", "Skin", "Frame with ROI", "Face pulse"]
    video_frames = add_labels_to_frames(video_frames, labels)
    pulse_extraction = stack_frames_horizontally(video_frames)
    duration_ms = (time.time() - start_time) * 1000
    add_txt_to_image(pulse_extraction, [f"FPS: {int(1000 / duration_ms)}"], loc="tc")
    cv2.imshow("Raw frame", cv2.cvtColor(pulse_extraction, cv2.COLOR_RGB2BGR))
    return {"rgb": global_rgb, "pulse_extraction": pulse_extraction}, prev_frames


def extract_rgb_from_skin(frame: np.ndarray, model: SegmentationModel, prev_frames: torch.Tensor):
    start_time = time.time()
    frame = avg_pool(frame, kernel=(4, 4))
    frame_h, frame_w = frame.shape[:2]

    seg_mask, output_mask, input_frame, resize_size, crop_coords = predict(
        frame, model, ["skin", "nose"]
    )
    frame_mask = inverse_processing_mask(output_mask, crop_coords, resize_size, (frame_h, frame_w))

    skin_mask = frame_mask == 1

    global_rgb = frame[skin_mask].mean(0)  # H x W x 3 -> 3

    prev_frames = prev_frames.roll(shifts=-1, dims=2)
    prev_frames[..., -1] = convolve_gaussian_2d(frame[..., 1], 27, 3, DEVICE)

    prev_skin_pixels = prev_frames[skin_mask]
    prev_skin_pixels = minmax(prev_skin_pixels, dim=-1, keepdim=True, scaler=255)
    last_frame_pixels = prev_skin_pixels[:, -1]
    last_frame_pixels = last_frame_pixels.squeeze().cpu().numpy().astype(np.uint8)
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
    duration_ms = (time.time() - start_time) * 1000
    add_txt_to_image(pulse_extraction, [f"FPS: {int(1000 / duration_ms)}"], loc="tc")
    cv2.imshow("Raw frame", cv2.cvtColor(pulse_extraction, cv2.COLOR_RGB2BGR))
    return {"rgb": global_rgb, "pulse_extraction": pulse_extraction}, prev_frames


def plot_signals(rgb: np.ndarray, pos: np.ndarray, fps: float, filename: str):
    def get_signal_frequencies(signal: np.ndarray, fs: float):
        y_fft = np.fft.fft(signal)  # Original FFT
        y_fft = y_fft[: round(len(signal) / 2)]  # First half ( pos freqs )
        y_fft = np.abs(y_fft)  # Absolute value of magnitudes
        y_fft = y_fft / max(y_fft)  # Normalized so max = 1
        freq_x_axis = np.linspace(0, fs / 2, len(y_fft))
        return freq_x_axis, y_fft

    sos = signal.cheby2(4, 10, [2 / 3, 4], "bandpass", output="sos", fs=fps)
    pos = signal.sosfiltfilt(sos, pos)

    freqs, pos_fft = get_signal_frequencies(pos, fps)
    duration = len(rgb) / fps
    t = np.linspace(0, duration, len(rgb))
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

    fig.savefig(filename, bbox_inches="tight")


def main():
    mode = "skin"  # skin or box
    filename = "temp/input/video_2.MOV"
    # filename = 2  # webcam

    model = load_model()
    pos_extractor = POSExtractor(FPS).to(DEVICE)

    if mode == "skin":
        video_w, video_h = get_video_size(filename)
        h, w = video_h // 4, video_w // 4
        process_fn = partial(extract_rgb_from_skin, model=model)
    else:
        h, w = 220, 130
        process_fn = partial(extract_rgb_from_box, model=model, box_size=(h, w))

    prev_frames = (torch.zeros(h, w, 90) + 0.5).to(DEVICE)

    result = process_video(process_fn, prev_frames, filename, start_frame=0, end_frame=-1)
    pulse_extraction = result["pulse_extraction"]
    save_frames_to_video(pulse_extraction, FPS, f"temp/pulse_extraction_from_{mode}.mp4")

    rgb = np.stack(result["rgb"])
    rgb_input = torch.from_numpy(rgb).unsqueeze(0).to(DEVICE).float()
    pos = pos_extractor(rgb_input).squeeze().cpu().numpy()

    plot_signals(rgb, pos, FPS, filename=f"temp/signals_from_{mode}.jpg")


if __name__ == "__main__":
    main()