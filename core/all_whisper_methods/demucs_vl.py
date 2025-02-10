import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import torch
from rich.console import Console
from rich import print as rprint
from demucs.pretrained import get_model
from demucs.audio import save_audio
from torch.cuda import is_available as is_cuda_available
from typing import Optional
from demucs.api import Separator
from demucs.apply import BagOfModels
import gc

"""
使用 Demucs（一个基于深度学习的音频分离模型）将音频文件 分离为人声（vocal）和背景音乐（background）
Demucs 是一种基于深度学习的音频分离模型，这个脚本适用于需要 提取人声或伴奏 的应用，如：
    KTV 伴奏提取
    播客/采访去除背景噪音
    音乐混音与采样
"""

AUDIO_DIR = "output/audio"
RAW_AUDIO_FILE = os.path.join(AUDIO_DIR, "raw.mp3") #  原始音频文件路径（待处理）
BACKGROUND_AUDIO_FILE = os.path.join(AUDIO_DIR, "background.mp3") #  背景音乐文件路径（处理后）
VOCAL_AUDIO_FILE = os.path.join(AUDIO_DIR, "vocal.mp3") #  人声文件路径（处理后）

class PreloadedSeparator(Separator):
    """
    继承 Demucs 的 Separator 类，封装 Demucs 处理逻辑。
    自动检测设备：
        cuda（NVIDIA 显卡）
        mps（Apple M1/M2）
        cpu（默认）
    shifts, overlap, split, segment:这些参数用于控制音频分离的精度和平滑度，默认值可以保持不变。
    """
    def __init__(self, model: BagOfModels, shifts: int = 1, overlap: float = 0.25,
                 split: bool = True, segment: Optional[int] = None, jobs: int = 0):
        self._model, self._audio_channels, self._samplerate = model, model.audio_channels, model.samplerate
        device = "cuda" if is_cuda_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.update_parameter(device=device, shifts=shifts, overlap=overlap, split=split,
                            segment=segment, jobs=jobs, progress=True, callback=None, callback_arg=None)

def demucs_main():
    if os.path.exists(VOCAL_AUDIO_FILE) and os.path.exists(BACKGROUND_AUDIO_FILE):
        rprint(f"[yellow]⚠️ {VOCAL_AUDIO_FILE} and {BACKGROUND_AUDIO_FILE} already exist, skip Demucs processing.[/yellow]")
        return
    
    # 加载模型并执行音频分离
    console = Console()
    os.makedirs(AUDIO_DIR, exist_ok=True)
    
    console.print("🤖 Loading <htdemucs> model...")
    model = get_model('htdemucs')
    separator = PreloadedSeparator(model=model, shifts=1, overlap=0.25)
    
    # 通过 Demucs 模型分离音频，outputs 是一个字典，包含：outputs["vocals"]：人声; outputs["bass"], outputs["drums"], outputs["other"]（背景音）
    console.print("🎵 Separating audio...")
    _, outputs = separator.separate_audio_file(RAW_AUDIO_FILE)
    
    kwargs = {"samplerate": model.samplerate, "bitrate": 64, "preset": 2, 
             "clip": "rescale", "as_float": False, "bits_per_sample": 16}
    
    # 将 vocals（人声）保存到 vocal.mp3。
    console.print("🎤 Saving vocals track...")
    save_audio(outputs['vocals'].cpu(), VOCAL_AUDIO_FILE, **kwargs)
    
    # 通过求和非人声音轨来构造背景音，并保存到 background.mp3。
    console.print("🎹 Saving background music...")
    background = sum(audio for source, audio in outputs.items() if source != 'vocals')
    save_audio(background.cpu(), BACKGROUND_AUDIO_FILE, **kwargs)
    
    # Clean up memory
    del outputs, background, model, separator
    gc.collect()
    
    console.print("[green]✨ Audio separation completed![/green]")

if __name__ == "__main__":
    demucs_main()
