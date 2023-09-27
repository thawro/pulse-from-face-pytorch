import torch
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def filter_pos(pos: np.ndarray, fps: float):
    sos = signal.cheby2(4, 10, [2 / 3, 4], "bandpass", output="sos", fs=fps)
    return signal.sosfiltfilt(sos, pos)


def plot_signals(rgb: np.ndarray, pos: np.ndarray, fps: float, filepath: str):
    def get_signal_frequencies(signal: np.ndarray, fs: float):
        y_fft = np.fft.fft(signal)  # Original FFT
        y_fft = y_fft[: round(len(signal) / 2)]  # First half ( pos freqs )
        y_fft = np.abs(y_fft)  # Absolute value of magnitudes
        y_fft = y_fft / max(y_fft)  # Normalized so max = 1
        freq_x_axis = np.linspace(0, fs / 2, len(y_fft))
        return freq_x_axis, y_fft

    pos = filter_pos(pos, fps)

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

    axes[0].set_xlabel("Time [s]")
    axes[1].set_xlabel("Time [s]")
    axes[2].set_xlabel("Frequency [Hz]")
    axes[2].set_ylabel("Magnitude")

    fig.savefig(filepath, bbox_inches="tight")


class POSExtractor(torch.nn.Module):
    def __init__(self, fps: float):
        super().__init__()
        win_len_sec = 1.6
        win_len = int(win_len_sec * fps)
        self.s_proj = torch.nn.Linear(3, 2)
        with torch.no_grad():
            M = torch.tensor([[0, 1, -1], [-2, 1, 1]], dtype=torch.float)
            self.s_proj.weight = torch.nn.Parameter(M)

        self.win_len = win_len

    def forward(self, rgbs):
        B, N, C = rgbs.shape
        all_rgbs = []
        for _rgb in rgbs:
            _rgb = _rgb.unfold(dimension=0, size=self.win_len, step=1)
            all_rgbs.append(_rgb)
        all_rgbs = torch.stack(all_rgbs).permute(0, 1, 3, 2)
        rgb = all_rgbs.squeeze(1)  # .flatten(0, 1)
        if len(rgb.shape) == 4:
            rgb = rgb.squeeze(0)
        fold_B, _, C = rgb.shape
        H = torch.zeros(B, N).to(rgbs.device)
        Cn = rgb / rgb.mean(dim=1, keepdim=True)

        S = self.s_proj(Cn.flatten(0, 1)).reshape(fold_B, self.win_len, 2).permute(0, 2, 1)

        std_0 = S[:, 0].std(1)
        std_1 = S[:, 1].std(1)

        h = S[:, 0] + (std_0 / std_1).unsqueeze(1) * S[:, 1]
        h = h - h.mean(1, keepdim=True)
        h = h.detach()

        i = 0
        for b in range(B):
            if N == self.win_len:
                H[b] = h[b]
            else:
                for m in range(0, N - self.win_len):
                    n = m + self.win_len
                    H[b, m:n] += h[i]
                    i += 1
        return H
