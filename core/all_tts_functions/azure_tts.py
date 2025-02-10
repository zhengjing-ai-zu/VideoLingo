import requests
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from core.config_utils import load_key

"""
微软 Azure 语音合成 API。
"""
def azure_tts(text: str, save_path: str) -> None:
    """
    参数
        text：要转换为语音的文本。
        save_path：输出音频文件的路径。
    
    该脚本 调用 Azure TTS API，将文本转换为语音，并保存为 WAV 文件。
    使用 SSML 控制合成语音的参数（语音模型、语言等）。
    支持不同的 TTS 语音模型（如 zh-CN-XiaoxiaoNeural）。
    适用于批量生成音频字幕，可能用于 视频翻译+配音。
    
    """
    url = "https://api.302.ai/cognitiveservices/v1"
    
    API_KEY = load_key("azure_tts.api_key")
    voice = load_key("azure_tts.voice")
    
    payload = f"""<speak version='1.0' xml:lang='zh-CN'>
    <voice name='{voice}'>
        {text}
    </voice>    
</speak>"""

    headers = {
       'Authorization': f'Bearer {API_KEY}',
       'X-Microsoft-OutputFormat': 'riff-16khz-16bit-mono-pcm',
       'Content-Type': 'application/ssml+xml'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    with open(save_path, 'wb') as f:
        f.write(response.content)
    print(f"Audio saved to {save_path}")

if __name__ == "__main__":
    azure_tts("Hi! Welcome to VideoLingo!", "test.wav")