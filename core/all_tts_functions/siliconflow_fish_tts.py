import requests
from pathlib import Path
import os, sys
import base64
import uuid
from typing import List, Tuple
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from core.config_utils import load_key, update_key
from core.step1_ytdlp import find_video_files
from core.all_whisper_methods.audio_preprocess import get_audio_duration
import hashlib
from rich import print as rprint
from pydub import AudioSegment
import time
from rich.panel import Panel
from rich.text import Text

"""
主要功能是调用 SiliconFlow Fish TTS API，将文本转换为语音，并支持三种模式：
    Preset（预设模式）：使用默认的 fishaudio/fish-speech-1.4 语音模型。
    Custom（自定义模式）：通过上传参考音频，创建自定义声音，并使用它进行文本合成。
    Dynamic（动态模式）：使用参考音频和文本进行即时语音合成，而不创建新的自定义声音。
支持音频合并
支持 VideoLingo 场景
使用 requests 发送 API 请求
使用 pydub 处理音频
通过 rich 打印美观的日志
"""

API_URL_SPEECH = "https://api.siliconflow.cn/v1/audio/speech"
API_URL_VOICE = "https://api.siliconflow.cn/v1/uploads/audio/voice"

AUDIO_REFERS_DIR = "output/audio/refers"
MODEL_NAME = "fishaudio/fish-speech-1.4"
REFER_MAX_LENGTH = 90

def _get_headers():
    return {"Authorization": f'Bearer {load_key("sf_fish_tts.api_key")}', "Content-Type": "application/json"}

def siliconflow_fish_tts(text, save_path, mode="preset", voice_id=None, ref_audio=None, ref_text=None, check_duration=False):
    """向 SiliconFlow Fish TTS API 发送请求，将 text 转换为 .wav 音频，并保存到 save_path。"""
    
    sf_fish_set, headers = load_key("sf_fish_tts"), _get_headers()
    payload = {"model": MODEL_NAME, "response_format": "wav", "stream": False, "input": text}
    
    if mode == "preset": 
        payload["voice"] = f"fishaudio/fish-speech-1.4:{sf_fish_set['voice']}"
    elif mode == "custom": 
        if not voice_id: 
            raise ValueError("custom mode requires voice_id")
        payload["voice"] = voice_id
    elif mode == "dynamic":
        if not ref_audio or not ref_text: 
            raise ValueError("dynamic mode requires ref_audio and ref_text")
        with open(ref_audio, 'rb') as f: 
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')
        payload = {
            "model": MODEL_NAME,
            "response_format": "wav",
            "stream": False,
            "input": text,
            "voice": None,
            "references": [{
                "audio": f"data:audio/wav;base64,{audio_base64}",
                "text": ref_text
            }]
        }
    else: raise ValueError("Invalid mode")

    max_retries = 2
    retry_delay = 1
    
    for attempt in range(max_retries):
        response = requests.post(API_URL_SPEECH, json=payload, headers=headers)
        if response.status_code == 200:
            wav_file_path = Path(save_path).with_suffix('.wav')
            wav_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(wav_file_path, 'wb') as f: f.write(response.content)
            
            if check_duration:
                duration = get_audio_duration(wav_file_path)
                rprint(f"[blue]Audio Duration: {duration:.2f} seconds")
                
            rprint(f"[green]Successfully generated audio file: {wav_file_path}")
            return True
            
        error_msg = response.json()
        rprint(f"[red]Failed to generate audio | HTTP {response.status_code} (Attempt {attempt + 1}/{max_retries})")
        rprint(f"[red]Text: {text}")
        rprint(f"[red]Error details: {error_msg}")
        
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
            rprint(f"[yellow]Retrying in {retry_delay} second...")
            
    return False

def create_custom_voice(audio_path, text, custom_name=None):
    """
    该函数用于 创建自定义语音，即：
        上传一段音频
        配套文本
        获取一个 voice_id，用于后续 TTS 请求
    """
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found at {audio_path}")
    
    try:
        audio_base64 = f"data:audio/wav;base64,{base64.b64encode(open(audio_path, 'rb').read()).decode('utf-8')}"
        rprint(f"[yellow]✅ Successfully encoded audio file")
    except Exception as e:
        rprint(f"[red]❌ Error reading file: {str(e)}")
        raise
    
    payload = {
        "audio": audio_base64,
        "model": MODEL_NAME,
        "customName": custom_name or str(uuid.uuid4())[:8],
        "text": text
    }
    
    rprint(f"[yellow]🚀 Sending request to create voice...")
    response = requests.post(API_URL_VOICE, json=payload, headers=_get_headers())
    response_json = response.json()
    
    if response.status_code == 200:
        voice_id = response_json.get('uri')
        status_text = Text()
        status_text.append("✨ Successfully created custom voice!\n", style="green")
        status_text.append(f"🎙️ Voice ID: {voice_id}\n", style="green")
        status_text.append(f"⌛ Creation Time: {time.strftime('%Y-%m-%d %H:%M:%S')}", style="green")
        rprint(Panel(status_text, title="Voice Creation Status"))
        return voice_id
        
    error_text = Text()
    error_text.append("❌ Failed to create custom voice\n", style="red")
    error_text.append(f"⚠️ HTTP Status: {response.status_code}\n", style="red")
    error_text.append(f"💬 Error Details: {response_json}", style="red")
    rprint(Panel(error_text, title="Error", border_style="red"))
    raise ValueError(f"Failed to create custom voice 🚫 HTTP {response.status_code}, Error details: {response_json}")

def merge_audio(files: List[str], output: str) -> bool:
    """
    Merge audio files, add a brief silence
    用途：将多个 .wav 文件合并，之间加入 100ms 的静音。
    """
    try:
        # Create an empty audio segment
        combined = AudioSegment.empty()
        silence = AudioSegment.silent(duration=100)  # 100ms silence
        
        # Add audio files one by one
        for file in files:
            audio = AudioSegment.from_wav(file)
            combined += audio + silence
        
        # Export the combined file
        combined.export(output, format="wav", parameters=[
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "1"
        ])
        
        if os.path.getsize(output) == 0:
            rprint(f"[red]Output file size is 0")
            return False
            
        rprint(f"[green]Successfully merged audio files")
        return True
        
    except Exception as e:
        rprint(f"[red]Failed to merge audio: {str(e)}")
        return False

def get_ref_audio(task_df) -> Tuple[str, str]:
    """
    Get reference audio and text, ensuring the combined text length does not exceed 100 characters
    从 task_df 选择一段音频，并确保文本长度 不超过 90 个字符，用于 动态模式 语音合成。
    """
    rprint(f"[blue]🎯 Starting reference audio selection process...")
    
    duration = 0
    selected = []
    combined_text = ""
    found_first = False
    
    for _, row in task_df.iterrows():
        current_text = row['origin']
        
        # If no valid record has been found yet
        if not found_first:
            if len(current_text) <= REFER_MAX_LENGTH:
                selected.append(row)
                combined_text = current_text
                duration += row['duration']
                found_first = True
                rprint(f"[yellow]📝 Found first valid row: {current_text[:50]}...")
            else:
                rprint(f"[yellow]⏭️ Skipping long row: {current_text[:50]}... ({len(current_text)} chars)")
            continue
            
        # Check subsequent rows
        new_text = combined_text + " " + current_text
        if len(new_text) > REFER_MAX_LENGTH:
            break
            
        selected.append(row)
        combined_text = new_text
        duration += row['duration']
        rprint(f"[yellow]📝 Added row: {current_text[:50]}...")
        
        if duration > 10:
            break
    
    if not selected:
        rprint(f"[red]❌ No valid segments found (all texts exceed {REFER_MAX_LENGTH} characters)")
        return None, None
        
    rprint(f"[blue]📊 Selected {len(selected)} segments, total duration: {duration:.2f}s")
    
    audio_files = [f"{AUDIO_REFERS_DIR}/{row['number']}.wav" for row in selected]
    rprint(f"[yellow]🎵 Audio files to merge: {audio_files}")
    
    combined_audio = f"{AUDIO_REFERS_DIR}/combined_reference.wav"
    success = merge_audio(audio_files, combined_audio)
    
    if not success:
        rprint(f"[red]❌ Error: Failed to merge audio files")
        return None, None
        
    rprint(f"[green]✅ Successfully created combined audio: {combined_audio}")
    rprint(f"[green]📝 Final combined text: {combined_text} | Length: {len(combined_text)}")
    
    return combined_audio, combined_text

def siliconflow_fish_tts_for_videolingo(text, save_as, number, task_df):
    sf_fish_set = load_key("sf_fish_tts")
    MODE = sf_fish_set["mode"]

    if MODE == "preset":
        return siliconflow_fish_tts(text, save_as, mode="preset")
    elif MODE == "custom":
        video_file = find_video_files()
        custom_name = hashlib.md5(video_file.encode()).hexdigest()[:8]
        rprint(f"[yellow]Using custom name: {custom_name}")
        log_name = load_key("sf_fish_tts.custom_name")
        
        if log_name != custom_name:
            # Get the merged reference audio and text
            ref_audio, ref_text = get_ref_audio(task_df)
            if ref_audio is None or ref_text is None:
                rprint(f"[red]Failed to get reference audio and text, falling back to preset mode")
                return siliconflow_fish_tts(text, save_as, mode="preset")
                
            voice_id = create_custom_voice(ref_audio, ref_text, custom_name)
            update_key("sf_fish_tts.voice_id", voice_id)
            update_key("sf_fish_tts.custom_name", custom_name)
        else:
            voice_id = load_key("sf_fish_tts.voice_id")
        return siliconflow_fish_tts(text=text, save_path=save_as, mode="custom", voice_id=voice_id)
    elif MODE == "dynamic":
        ref_audio_path = f"{AUDIO_REFERS_DIR}/{number}.wav"
        if not Path(ref_audio_path).exists():
            rprint(f"[red]Reference audio not found: {ref_audio_path}, falling back to preset mode")
            return siliconflow_fish_tts(text, save_as, mode="preset")
            
        ref_text = task_df[task_df['number'] == number]['origin'].iloc[0]
        return siliconflow_fish_tts(text=text, save_path=save_as, mode="dynamic", ref_audio=str(ref_audio_path), ref_text=ref_text)
    else:
        raise ValueError("Invalid mode. Choose 'preset', 'custom', or 'dynamic'")

if __name__ == '__main__':
    pass
    # create_custom_voice("output/audio/refers/1.wav", "Okay folks, welcome back. This is price action model number four, position trading.")
    siliconflow_fish_tts("가을 나뭇잎이 부드럽게 떨어지는 생생한 색깔을 주목하지 않을 수 없었다", "preset_test.wav", mode="preset", check_duration=True)
    # siliconflow_fish_tts("使用客制化音色测试", "custom_test.wav", mode="custom", voice_id="speech:your-voice-name:cm04pf7az00061413w7kz5qxs:mjtkgbyuunvtybnsvbxd")
    # siliconflow_fish_tts("使用动态音色测试", "dynamic_test.wav", mode="dynamic", ref_audio="output/audio/refers/1.wav", ref_text="Okay folks, welcome back. This is price action model number four, position trading.")