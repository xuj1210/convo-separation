# audio_preprocess.py

import torch
import torchaudio
from noisereduce.torchgate import TorchGate as TG


def preprocess_audio(audio_file_path: str, target_sample_rate: int = 16000, target_peak_level: float = 0.95, as_mono=True):
    """
    Loads an audio file, converts it to mono, normalizes its volume, and resamples it.

    Args:
        audio_file_path (str): The path to the input audio file.
        target_sample_rate (int): The desired sample rate for the output waveform.
        target_peak_level (float): The target peak level for volume normalization (e.g., 0.95).

    Returns:
        tuple[torch.Tensor, int]: A tuple containing the processed waveform and its sample rate.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    
    waveform, sample_rate = torchaudio.load(audio_file_path)
    waveform = waveform.to(device)

    if as_mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate

    # tg = TG(
    #         sr=target_sample_rate, 
    #         nonstationary=True,
    #         prop_decrease=0.25,
    #         n_std_thresh_stationary=2
    #     ).to(device)
    # waveform = tg(waveform)

    peak_value = torch.max(torch.abs(waveform))
    if peak_value > 0:
        scaling_factor = target_peak_level / peak_value
        waveform *= scaling_factor
    
    save = waveform.detach().cpu()
    torchaudio.save('output/processed_audio.wav', save, sample_rate)

    
    return waveform, sample_rate