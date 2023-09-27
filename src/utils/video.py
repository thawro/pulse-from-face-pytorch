from moviepy.editor import ImageSequenceClip
import numpy as np
import cv2
from typing import Callable, Any
from collections import defaultdict
import torch


def save_frames_to_video(frames: list[np.ndarray], fps: int, filepath: str):
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(filepath, fps=fps)


def record_webcam_to_mp4(filename: str = "video.mp4"):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cv2.imshow("Frame", frame)
        c = cv2.waitKey(1)
        if c == 27:
            break

    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(filename)

    cap.release()
    cv2.destroyAllWindows()


def get_video_params(filename: str) -> dict[str, int | float]:
    cap = cv2.VideoCapture(filename)
    w = int(cap.get(3))
    h = int(cap.get(4))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    cv2.destroyAllWindows()
    return {"width": w, "height": h, "frame_count": frame_count}


def process_video(
    processing_fn: Callable[[np.ndarray], dict[str, Any]],
    prev_frames: torch.Tensor,
    pos_signal: torch.Tensor,
    rgb_signal: torch.Tensor,
    filename: str | int = "video.mp4",
    start_frame: int = 0,
    end_frame: int = -1,
    verbose: bool = False,
) -> dict[str, list[Any]]:
    cap = cv2.VideoCapture(filename)
    if end_frame == -1:
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = {"prev_frames": prev_frames, "pos_signal": pos_signal, "rgb_signal": rgb_signal}
    results = defaultdict(lambda: [], {})
    idx = 0
    while True:
        idx += 1
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if verbose:
            print(idx)
        if idx < start_frame:
            continue
        if idx == end_frame:
            break
        if ret:
            result, out = processing_fn(frame=frame_rgb, idx=idx, **out)
            for name, r in result.items():
                results[name].append(r)
            if cv2.waitKey(1) == 27:
                break
    cap.release()
    cv2.destroyAllWindows()
    return results
