<div align="center">

<img src="/docs/logo.png" alt="VideoLingo Logo" height="140">

# Connect the World, Frame by Frame

<a href="https://trendshift.io/repositories/12200" target="_blank"><img src="https://trendshift.io/api/badge/repositories/12200" alt="Huanshere%2FVideoLingo | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

[**English**](/README.md)｜[**简体中文**](/translations/README.zh.md)｜[**繁體中文**](/translations/README.zh-TW.md)｜[**日本語**](/translations/README.ja.md)｜[**Español**](/translations/README.es.md)｜[**Русский**](/translations/README.ru.md)｜[**Français**](/translations/README.fr.md)

</div>

## 🌟 Overview ([Try VL Now!](https://videolingo.io))

VideoLingo is an all-in-one video translation, localization, and dubbing tool aimed at generating Netflix-quality subtitles. It eliminates stiff machine translations and multi-line subtitles while adding high-quality dubbing, enabling global knowledge sharing across language barriers.

Key features:
- 🎥 YouTube video download via yt-dlp

- **🎙️ Word-level and Low-illusion subtitle recognition with WhisperX**

- **📝 NLP and AI-powered subtitle segmentation**

- **📚 Custom + AI-generated terminology for coherent translation**

- **🔄 3-step Translate-Reflect-Adaptation for cinematic quality**

- **✅ Netflix-standard, Single-line subtitles Only**

- **🗣️ Dubbing with GPT-SoVITS, Azure, OpenAI, and more**

- 🚀 One-click startup and processing in Streamlit

- 🌍 Multi-language support in Streamlit UI

- 📝 Detailed logging with progress resumption

Difference from similar projects: **Single-line subtitles only, superior translation quality, seamless dubbing experience**

## 🎥 Demo

<table>
<tr>
<td width="50%">

### Russian Translation
---
https://github.com/user-attachments/assets/25264b5b-6931-4d39-948c-5a1e4ce42fa7

</td>
<td width="50%">

### GPT-SoVITS Dubbing
---
https://github.com/user-attachments/assets/47d965b2-b4ab-4a0b-9d08-b49a7bf3508c

</td>
</tr>
</table>

### Language Support

**Input Language Support(more to come):**

🇺🇸 English 🤩 | 🇷🇺 Russian 😊 | 🇫🇷 French 🤩 | 🇩🇪 German 🤩 | 🇮🇹 Italian 🤩 | 🇪🇸 Spanish 🤩 | 🇯🇵 Japanese 😐 | 🇨🇳 Chinese* 😊

> *Chinese uses a separate punctuation-enhanced whisper model, for now...

**Translation supports all languages, while dubbing language depends on the chosen TTS method.**

## Installation

You don't have to read the whole docs, [**here**](https://share.fastgpt.in/chat/share?shareId=066w11n3r9aq6879r4z0v9rh) is an online AI agent to help you.

> **Note:** For Windows users with NVIDIA GPU, follow these steps before installation:
> 1. Install [CUDA Toolkit 12.6](https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.76_windows.exe)
> 2. Install [CUDNN 9.3.0](https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn_9.3.0_windows.exe)
> 3. Add `C:\Program Files\NVIDIA\CUDNN\v9.3\bin\12.6` to your system PATH
> 4. Restart your computer

> **Note:** FFmpeg is required. Please install it via package managers:
> - Windows: ```choco install ffmpeg``` (via [Chocolatey](https://chocolatey.org/))
> - macOS: ```brew install ffmpeg``` (via [Homebrew](https://brew.sh/))
> - Linux: ```sudo apt install ffmpeg``` (Debian/Ubuntu)

1. Clone the repository

```bash
git clone https://github.com/Huanshere/VideoLingo.git
cd VideoLingo
```

2. Install dependencies(requires `python=3.10`)

```bash
conda create -n videolingo python=3.10.0 -y
conda activate videolingo
python install.py
```

3. Start the application

```bash
streamlit run st.py
```

### Docker
Alternatively, you can use Docker (requires CUDA 12.4 and NVIDIA Driver version >550), see [Docker docs](/docs/pages/docs/docker.en-US.md):

```bash
docker build -t videolingo .
docker run -d -p 8501:8501 --gpus all videolingo
```

## APIs
VideoLingo supports OpenAI-Like API format and various TTS interfaces:
- LLM: `claude-3-5-sonnet-20240620`, `deepseek-chat(v3)`, `gemini-2.0-flash-exp`, `gpt-4o`, ... (sorted by performance)
- WhisperX: Run whisperX locally or use 302.ai API
- TTS: `azure-tts`, `openai-tts`, `siliconflow-fishtts`, **`fish-tts`**, `GPT-SoVITS`, `edge-tts`, `*custom-tts`(You can modify your own TTS in custom_tts.py!)

> **Note:** VideoLingo works with **[302.ai](https://gpt302.saaslink.net/C2oHR9)** - one API key for all services (LLM, WhisperX, TTS). Or run locally with Ollama and Edge-TTS for free, no API needed!

For detailed installation, API configuration, and batch mode instructions, please refer to the documentation: [English](/docs/pages/docs/start.en-US.md) | [中文](/docs/pages/docs/start.zh-CN.md)

## Current Limitations

1. WhisperX transcription performance may be affected by video background noise, as it uses wav2vac model for alignment. For videos with loud background music, please enable Voice Separation Enhancement. Additionally, subtitles ending with numbers or special characters may be truncated early due to wav2vac's inability to map numeric characters (e.g., "1") to their spoken form ("one").

2. Using weaker models can lead to errors during intermediate processes due to strict JSON format requirements for responses. If this error occurs, please delete the `output` folder and retry with a different LLM, otherwise repeated execution will read the previous erroneous response causing the same error.

3. The dubbing feature may not be 100% perfect due to differences in speech rates and intonation between languages, as well as the impact of the translation step. However, this project has implemented extensive engineering processing for speech rates to ensure the best possible dubbing results.

4. **Multilingual video transcription recognition will only retain the main language**. This is because whisperX uses a specialized model for a single language when forcibly aligning word-level subtitles, and will delete unrecognized languages.

5. **Cannot dub multiple characters separately**, as whisperX's speaker distinction capability is not sufficiently reliable.

## 📄 License

This project is licensed under the Apache 2.0 License. Special thanks to the following open source projects for their contributions:

[whisperX](https://github.com/m-bain/whisperX), [yt-dlp](https://github.com/yt-dlp/yt-dlp), [json_repair](https://github.com/mangiucugna/json_repair), [BELLE](https://github.com/LianjiaTech/BELLE)

## 📬 Contact Me

- Submit [Issues](https://github.com/Huanshere/VideoLingo/issues) or [Pull Requests](https://github.com/Huanshere/VideoLingo/pulls) on GitHub
- DM me on Twitter: [@Huanshere](https://twitter.com/Huanshere)
- Email me at: team@videolingo.io

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Huanshere/VideoLingo&type=Timeline)](https://star-history.com/#Huanshere/VideoLingo&Timeline)

---

<p align="center">If you find VideoLingo helpful, please give me a ⭐️!</p>

## 项目理解（个人）

VideoLingo 是一个开源项目，旨在提供 Netflix 级别的视频字幕处理功能，包括字幕切割、翻译、对齐，甚至配音，实现一键全自动化处理。

### 项目结构：

- .streamlit/：Streamlit 配置文件。
- batch/：包含批处理脚本，用于处理批量视频文件。
- core/：核心功能模块，负责视频处理的主要逻辑。（目录是该项目的核心逻辑部分，包含多个子模块，主要涉及 文本转语音（TTS）、语音识别（Whisper）、文本处理（SpaCy） 以及完整的 视频翻译和合成流程。）
    * 下载视频 ➝ step1_ytdlp.py
    * 语音识别 ➝ step2_whisperX.py
    * 字幕优化 ➝ step3 & step4
    * 字幕时间轴调整 ➝ step5 & step6
    * 字幕合成 ➝ step7
    * 生成配音 ➝ step8 & step9
    * 合成最终视频 ➝ step10 ~ step12
    * 该项目支持 多种 TTS 方案、语音识别（WhisperX），并结合 GPT 进行优化，是一个完整的 视频翻译+配音自动化解决方案。
- docs/：项目的文档文件，提供使用说明和开发者指南。
- st_components/：Streamlit 组件，用于构建项目的前端界面。
- translations/：翻译相关的资源文件。
- Dockerfile：用于构建项目的 Docker 镜像，方便部署。
- OneKeyStart.bat：Windows 下的一键启动脚本。
- VideoLingo_colab.ipynb：Google Colab 的 Jupyter Notebook，方便在云端运行项目。
- config.yaml：项目的配置文件。（该文件包含应用程序的配置参数，如默认语言设置、翻译 API 密钥、音频处理参数等。开发者可以根据需要修改该文件，以适应不同的应用场景。）
- custom_terms.xlsx：自定义术语表，用于翻译时的术语管理。
- install.py：安装脚本，帮助用户快速配置环境。（这是一个安装脚本，用于帮助用户快速配置运行环境。它会检查系统的依赖项，安装所需的 Python 包，并进行必要的初始化操作。）
- requirements.txt：Python 依赖包列表。
- st.py：Streamlit 应用的主脚本，启动前端界面。（项目的主入口，使用 Streamlit 构建前端界面。该脚本负责加载配置文件（config.yaml），初始化应用程序，并定义用户交互界面。用户可以通过该界面上传视频文件，选择翻译语言，设置配音选项等。在用户提交任务后，st.py 会调用核心处理模块，执行视频的字幕提取、翻译、对齐和配音等操作。）

### 主要功能：

- 字幕切割：自动检测视频中的语音段落，并将其切割成独立的字幕片段。
- 字幕翻译：利用 AI 模型，将原始字幕翻译成目标语言。
- 字幕对齐：确保翻译后的字幕与视频中的语音同步。
- 自动配音：为翻译后的字幕生成对应的语音，并与视频合成，实现全自动配音。

### 使用方法：

用户可以通过运行 OneKeyStart.bat 脚本在 Windows 系统上一键启动应用，或使用 st.py 脚本启动 Streamlit 应用。在启动应用后，用户可以上传视频文件，配置翻译和配音选项，系统将自动处理并生成带有翻译字幕和配音的视频。

1. 环境配置：运行 install.py 脚本，或手动安装 requirements.txt 中列出的依赖项。
2. 配置文件：根据需要，修改 config.yaml 中的参数设置。
3. 启动应用：在命令行中运行 python st.py，启动 Streamlit 应用。
4. 上传视频：通过前端界面上传需要处理的视频文件，设置翻译和配音选项。
5. 处理结果：应用程序将自动处理视频，并生成带有翻译字幕和配音的输出视频。

### 技术栈：

Python：主要的编程语言。
Streamlit：用于构建前端界面。
Docker：用于容器化部署。
AI 模型：用于语音识别、翻译和语音合成。
项目的设计目标是简化视频翻译和本地化的流程，使用户能够高效地处理多语言视频内容。
