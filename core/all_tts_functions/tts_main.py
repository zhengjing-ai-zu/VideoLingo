import os, sys
import re
from rich import print as rprint
from pydub import AudioSegment

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from core.config_utils import load_key
from core.all_whisper_methods.audio_preprocess import get_audio_duration
from core.all_tts_functions.gpt_sovits_tts import gpt_sovits_tts_for_videolingo
from core.all_tts_functions.siliconflow_fish_tts import siliconflow_fish_tts_for_videolingo
from core.all_tts_functions.openai_tts import openai_tts
from core.all_tts_functions.fish_tts import fish_tts
from core.all_tts_functions.azure_tts import azure_tts
from core.all_tts_functions.edge_tts import edge_tts
from core.all_tts_functions.custom_tts import custom_tts
from core.ask_gpt import ask_gpt
from core.prompts_storage import get_correct_text_prompt

"""
TTS 的主要入口，将文本转换为音频（TTS，Text-to-Speech），并进行错误处理和音频校验。它支持多种 TTS 方法，并在失败时重试，确保最终生成有效的音频文件。
"""

def clean_text_for_tts(text):
    """Remove problematic characters for TTS"""
    chars_to_remove = ['&', '®', '™', '©']
    for char in chars_to_remove:
        text = text.replace(char, '')
    return text.strip()

def tts_main(text, save_as, number, task_df):
    text = clean_text_for_tts(text)
    # Check if text is empty or single character, single character voiceovers are prone to bugs
    cleaned_text = re.sub(r'[^\w\s]', '', text).strip()
    if not cleaned_text or len(cleaned_text) <= 1:
        silence = AudioSegment.silent(duration=100)  # 100ms = 0.1s
        silence.export(save_as, format="wav")
        rprint(f"Created silent audio for empty/single-char text: {save_as}")
        return
    
    # Skip if file exists
    if os.path.exists(save_as):
        return
    
    print(f"Generating <{text}...>")
    TTS_METHOD = load_key("tts_method")
    
    max_retries = 3
    for attempt in range(max_retries):
        # 允许最多 3 次重试。最后一次失败时，调用 GPT 纠正文本，以提高 TTS 成功率。
        try:
            if attempt >= max_retries - 1:
                print("Asking GPT to correct text...")
                correct_text = ask_gpt(get_correct_text_prompt(text),log_title='tts_correct_text')
                text = correct_text['text']
            if TTS_METHOD == 'openai_tts':
                openai_tts(text, save_as)
            elif TTS_METHOD == 'gpt_sovits':
                gpt_sovits_tts_for_videolingo(text, save_as, number, task_df)
            elif TTS_METHOD == 'fish_tts':
                fish_tts(text, save_as)
            elif TTS_METHOD == 'azure_tts':
                azure_tts(text, save_as)
            elif TTS_METHOD == 'sf_fish_tts':
                siliconflow_fish_tts_for_videolingo(text, save_as, number, task_df)
            elif TTS_METHOD == 'edge_tts':
                edge_tts(text, save_as)
            elif TTS_METHOD == 'custom_tts':
                custom_tts(text, save_as)
            
            # Check generated audio duration
            # 检查生成的音频时长：如果大于 0，说明音频正常，退出循环。如果等于 0：删除无效文件。最终失败后，生成 100ms 静音文件。
            duration = get_audio_duration(save_as)
            if duration > 0:
                break
            else:
                if os.path.exists(save_as):
                    os.remove(save_as)
                if attempt == max_retries - 1:
                    print(f"Warning: Generated audio duration is 0 for text: {text}")
                    # Create silent audio file
                    silence = AudioSegment.silent(duration=100)  # 100ms silence
                    silence.export(save_as, format="wav")
                    return
                print(f"Attempt {attempt + 1} failed, retrying...")
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to generate audio after {max_retries} attempts: {str(e)}")
            print(f"Attempt {attempt + 1} failed, retrying...")