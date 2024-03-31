import librosa
import torch
from torchaudio.transforms import Resample

def get_model():
    import whisper
    model = whisper.load_model("small", device='cpu')
    return model.encoder

resample = None
def get_content(model=None, src_path=None):
    global  resample

    wav, _ = librosa.load(src_path, sr=32000)
    device = model.parameters().__next__().device
    dtype = model.parameters().__next__().dtype
    if resample is None:
        resample = Resample(32000, 16000).to(device).to(dtype)
    wav_32k_tensor = torch.from_numpy(wav).to(device).to(dtype)[None]
    wav_16k_tensor = resample(wav_32k_tensor)[0]

    from whisper import log_mel_spectrogram, pad_or_trim
    mel = log_mel_spectrogram(wav_16k_tensor).to(device).to(dtype)[:, :3000]
    # if torch.cuda.is_available():
    #     mel = mel.to(torch.float16)
    feature_len = mel.shape[-1] // 2
    assert  mel.shape[-1] < 3000, "输入音频过长，只允许输入30以内音频"
    with torch.no_grad():
        feature = model(pad_or_trim(mel, 3000).unsqueeze(0).to(device).to(dtype))[:1, :feature_len, :].transpose(1,2)
    return feature

