# Mila Local Voice Tutor Desktop

这个版本使用 Tkinter 原生窗口，不会打开网页。语音识别和大语言模型在本地运行；老师说话时使用 `teacher_speaking.gif`，等待和听你说话时固定显示 `teacher.gif` 的第一帧。

## Windows 首次安装

1. 安装 Python 3.11 或 3.12，并勾选 **Add Python to PATH**。
2. 双击 `setup.bat`，等待依赖安装完成。
3. 打开 LM Studio，加载一个英文对话模型。
4. 在 **Developer / Local Server** 中启动服务器，默认地址应为 `http://127.0.0.1:1234/v1`。
5. 双击 `run.bat`。

首次启动会下载 `faster-whisper` 的 `base.en` 模型。以后启动直接使用本地缓存。

## 使用方法

- 点击“开始说英语”，或者按一次空格键开始录音。
- 说完后再点击按钮，或者再按一次空格键，程序会立刻识别并发送。
- 窗口同时显示：你的英文、你的中文翻译、老师的英文、老师的中文翻译。
- 老师的英文逐字出现；第一句生成完成后就开始朗读，不必等全部内容完成。
- 点击“停止老师”可以中断当前回答和朗读。

## 可选设置

在启动前可以用环境变量调整：

```powershell
$env:LOCAL_LLM_BASE_URL="http://127.0.0.1:1234/v1"
$env:LOCAL_LLM_MODEL="你在 LM Studio 中加载的模型名"
$env:WHISPER_MODEL_NAME="small.en"
$env:EDGE_TTS_EN_VOICE="en-US-JennyNeural"
& .\run.bat
```

`LOCAL_LLM_MODEL` 留空时，程序会自动使用 LM Studio `/v1/models` 返回的第一个已加载模型。

默认语音输出使用 `edge-tts`，声音自然、启动较快，但合成语音时需要联网。若要完全离线，可在 PowerShell 中运行：

```powershell
$env:TTS_PROVIDER="pyttsx3"
& .\run.bat
```

完全离线模式会使用 Windows 已安装的英语语音，音质取决于系统语音包。

## 建议模型

你的 Intel N100 / 16GB 电脑可以先用 1.5B～3B 的 Q4 模型，以获得更快的口语响应。对话质量优先时可尝试 7B Q4，但首句等待会明显增加。

## 项目文件

- `app.py`：桌面应用主程序
- `teacher.gif`：等待和听用户说话时使用的老师图像
- `teacher_speaking.gif`：老师朗读时使用的动态图像
- `requirements.txt`：Python 依赖
- `setup.bat`：Windows 首次安装
- `run.bat`：Windows 日常启动
