from __future__ import annotations

import asyncio
import json
import os
import queue
import re
import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk
from typing import Callable, Iterator

import edge_tts
import numpy as np
import pygame
import pyttsx3
import requests
import soundfile as sf
from faster_whisper import WhisperModel
from PIL import Image, ImageSequence, ImageTk


APP_DIR = Path(__file__).resolve().parent
IDLE_GIF = APP_DIR / "teacher.gif"
SPEAKING_GIF = APP_DIR / "teacher_speaking.gif"

LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "").strip()
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "base.en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
EDGE_TTS_VOICE = os.getenv("EDGE_TTS_EN_VOICE", "en-US-JennyNeural")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "edge-tts").lower()
SAMPLE_RATE = 16_000

SYSTEM_PROMPT = """You are Mila, a warm and concise English conversation teacher.
Hold a natural conversation suitable for an intermediate English learner.
Keep each reply to 1-3 short sentences so voice conversation remains fast.
If the learner makes an important English mistake, respond naturally first and then give one very brief correction.

Return exactly these three XML elements and no markdown:
<reply_en>Your natural English reply.</reply_en>
<user_zh>A faithful Simplified Chinese translation of the learner's latest English.</user_zh>
<reply_zh>A faithful Simplified Chinese translation of reply_en.</reply_zh>

Put reply_en first. Do not put Chinese inside reply_en."""


class UiEvents:
    """Thread-safe bridge from workers to Tk's main thread."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.items: queue.Queue[tuple[Callable, tuple]] = queue.Queue()
        self.root.after(25, self._drain)

    def call(self, func: Callable, *args) -> None:
        self.items.put((func, args))

    def _drain(self) -> None:
        try:
            while True:
                func, args = self.items.get_nowait()
                func(*args)
        except queue.Empty:
            pass
        self.root.after(25, self._drain)


class TeacherView(ttk.Label):
    def __init__(self, master: tk.Misc, width: int = 410, height: int = 410) -> None:
        super().__init__(master, anchor="center")
        self.target_size = (width, height)
        self.frames: list[ImageTk.PhotoImage] = []
        self.delays: list[int] = []
        self.frame_index = 0
        self.animation_job: str | None = None
        self._cache: dict[tuple[Path, bool], tuple[list[ImageTk.PhotoImage], list[int]]] = {}
        self.show_idle()

    def _load(self, path: Path, animated: bool) -> None:
        if not path.exists():
            self.configure(text=f"Missing image: {path.name}")
            return

        cache_key = (path, animated)
        cached = self._cache.get(cache_key)
        if cached:
            frames, delays = cached
        else:
            image = Image.open(path)
            source_frames = ImageSequence.Iterator(image) if animated else [image.copy()]
            frames = []
            delays = []
            for frame in source_frames:
                rgba = frame.convert("RGBA")
                rgba.thumbnail(self.target_size, Image.Resampling.LANCZOS)
                canvas = Image.new("RGBA", self.target_size, (244, 247, 252, 255))
                x = (self.target_size[0] - rgba.width) // 2
                y = (self.target_size[1] - rgba.height) // 2
                canvas.alpha_composite(rgba, (x, y))
                frames.append(ImageTk.PhotoImage(canvas))
                delays.append(max(30, int(frame.info.get("duration", image.info.get("duration", 50)))))
            self._cache[cache_key] = (frames, delays)

        self.frames = frames
        self.delays = delays
        self.frame_index = 0
        self.configure(image=self.frames[0], text="")
        if animated and len(self.frames) > 1:
            self._animate()

    def _cancel(self) -> None:
        if self.animation_job:
            self.after_cancel(self.animation_job)
            self.animation_job = None

    def _animate(self) -> None:
        if not self.frames:
            return
        self.configure(image=self.frames[self.frame_index])
        delay = self.delays[self.frame_index]
        self.frame_index = (self.frame_index + 1) % len(self.frames)
        self.animation_job = self.after(delay, self._animate)

    def show_idle(self) -> None:
        self._cancel()
        self._load(IDLE_GIF, animated=False)

    def show_speaking(self) -> None:
        self._cancel()
        self._load(SPEAKING_GIF, animated=True)

    def preload_speaking(self) -> None:
        """Decode animation once during startup, then keep the idle image visible."""
        if (SPEAKING_GIF, True) not in self._cache:
            self._load(SPEAKING_GIF, animated=True)
            self.show_idle()


class Recorder:
    def __init__(self) -> None:
        self.stream = None
        self.blocks: list[np.ndarray] = []
        self.started_at = 0.0

    @property
    def recording(self) -> bool:
        return self.stream is not None

    def start(self) -> None:
        if self.recording:
            return
        # Import lazily so the program can still show a useful window/error on a
        # machine where PortAudio or the microphone driver is unavailable.
        import sounddevice as sd

        self.blocks = []
        self.started_at = time.monotonic()

        def callback(indata, frames, time_info, status) -> None:
            if status:
                print("[microphone]", status)
            self.blocks.append(indata.copy())

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=1024,
            callback=callback,
        )
        self.stream.start()

    def stop(self) -> Path | None:
        if not self.stream:
            return None
        stream, self.stream = self.stream, None
        stream.stop()
        stream.close()
        if not self.blocks or time.monotonic() - self.started_at < 0.25:
            return None
        audio = np.concatenate(self.blocks, axis=0)
        path = Path(tempfile.mkstemp(prefix="voice_tutor_input_", suffix=".wav")[1])
        sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
        return path


class LocalLLM:
    def __init__(self, base_url: str, requested_model: str = "") -> None:
        self.base_url = base_url
        self.requested_model = requested_model
        self._resolved_model = ""

    def model(self) -> str:
        if self.requested_model:
            return self.requested_model
        if self._resolved_model:
            return self._resolved_model
        response = requests.get(f"{self.base_url}/models", timeout=5)
        response.raise_for_status()
        models = response.json().get("data") or []
        if not models:
            raise RuntimeError("LM Studio 没有已加载的模型。")
        self._resolved_model = str(models[0]["id"])
        return self._resolved_model

    def stream(self, messages: list[dict[str, str]], stop_event: threading.Event) -> Iterator[str]:
        payload = {
            "model": self.model(),
            "messages": messages,
            "temperature": 0.65,
            "max_tokens": 260,
            "stream": True,
        }
        with requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            stream=True,
            timeout=(5, 120),
        ) as response:
            response.raise_for_status()
            for raw in response.iter_lines(decode_unicode=True):
                if stop_event.is_set():
                    return
                if not raw:
                    continue
                line = raw.strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    return
                try:
                    data = json.loads(line)
                    token = data["choices"][0].get("delta", {}).get("content", "")
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue
                if token:
                    yield token


def tag_value(text: str, tag: str, allow_partial: bool = True) -> str:
    start_marker = f"<{tag}>"
    end_marker = f"</{tag}>"
    start = text.find(start_marker)
    if start < 0:
        return ""
    start += len(start_marker)
    end = text.find(end_marker, start)
    if end < 0:
        return text[start:].strip() if allow_partial else ""
    return text[start:end].strip()


def complete_sentences(text: str, spoken_chars: int, force: bool = False) -> tuple[list[str], int]:
    remaining = text[spoken_chars:]
    if not remaining:
        return [], spoken_chars
    matches = list(re.finditer(r".*?[.!?](?:[\"']|\s|$)", remaining, flags=re.S))
    if matches:
        consumed = matches[-1].end()
        chunks = [m.group(0).strip() for m in matches if m.group(0).strip()]
        return chunks, spoken_chars + consumed
    if force and remaining.strip():
        return [remaining.strip()], len(text)
    return [], spoken_chars


class VoiceTutorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Mila - 本地英语口语老师")
        self.root.geometry("1000x900")
        self.root.minsize(820, 720)
        self.root.configure(bg="#eef2f8")

        self.events = UiEvents(root)
        self.recorder = Recorder()
        self.llm = LocalLLM(LOCAL_LLM_BASE_URL, LOCAL_LLM_MODEL)
        self.whisper: WhisperModel | None = None
        self.history: list[dict[str, str]] = []
        self.stop_event = threading.Event()
        self.busy = False
        self.tts_queue: queue.Queue[str | None] = queue.Queue()
        self.tts_worker_event: threading.Event | None = None

        self._build_ui()
        self.root.after(150, self.teacher.preload_speaking)
        self.root.bind("<space>", self._toggle_from_key)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        threading.Thread(target=self._warm_up, daemon=True).start()

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"), foreground="#17345c", background="#eef2f8")
        style.configure("Status.TLabel", font=("Segoe UI", 10), foreground="#426083", background="#eef2f8")
        style.configure("Talk.TButton", font=("Microsoft YaHei UI", 13, "bold"), padding=(16, 12))

        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Mila · Local English Voice Tutor", style="Title.TLabel").pack()
        self.status_var = tk.StringVar(value="正在准备语音识别模型……")
        ttk.Label(outer, textvariable=self.status_var, style="Status.TLabel").pack(pady=(2, 8))

        self.teacher = TeacherView(outer, 400, 400)
        self.teacher.pack()

        subtitles = ttk.Frame(outer)
        subtitles.pack(fill="both", expand=True, pady=(8, 6))
        subtitles.columnconfigure(0, weight=1)
        subtitles.rowconfigure(0, weight=1)

        self.transcript = scrolledtext.ScrolledText(
            subtitles,
            height=13,
            wrap="word",
            font=("Microsoft YaHei UI", 11),
            bg="#ffffff",
            fg="#1d2a3a",
            relief="flat",
            padx=14,
            pady=10,
        )
        self.transcript.grid(row=0, column=0, sticky="nsew")
        self.transcript.tag_configure("user_head", foreground="#2470c7", font=("Microsoft YaHei UI", 10, "bold"))
        self.transcript.tag_configure("teacher_head", foreground="#a44b16", font=("Microsoft YaHei UI", 10, "bold"))
        self.transcript.tag_configure("english", foreground="#172234", font=("Segoe UI", 12))
        self.transcript.tag_configure("chinese", foreground="#58677c", font=("Microsoft YaHei UI", 10))
        self.transcript.configure(state="disabled")

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(6, 0))
        self.talk_button = ttk.Button(controls, text="🎙 开始说英语（空格键）", style="Talk.TButton", command=self.toggle_recording)
        self.talk_button.pack(side="left", fill="x", expand=True)
        self.stop_button = ttk.Button(controls, text="停止老师", command=self.stop_current)
        self.stop_button.pack(side="left", padx=(10, 0))
        self.clear_button = ttk.Button(controls, text="清空对话", command=self.clear_history)
        self.clear_button.pack(side="left", padx=(10, 0))

    def _warm_up(self) -> None:
        try:
            model = WhisperModel(
                WHISPER_MODEL_NAME,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
                cpu_threads=max(1, min(8, (os.cpu_count() or 4) - 1)),
                num_workers=1,
            )
            self.whisper = model
            model_name = self.llm.model()
            self.events.call(self._set_ready, f"准备完成 · Whisper {WHISPER_MODEL_NAME} · 本地模型 {model_name}")
        except Exception as exc:
            self.events.call(self._set_ready, f"准备未完成：{exc}")

    def _set_ready(self, text: str) -> None:
        self.status_var.set(text)

    def _toggle_from_key(self, _event) -> str:
        self.toggle_recording()
        return "break"

    def toggle_recording(self) -> None:
        if self.recorder.recording:
            self.finish_recording()
            return
        if self.busy:
            self.status_var.set("老师正在处理；可先点击“停止老师”再开始下一轮。")
            return
        try:
            self.recorder.start()
            self.teacher.show_idle()
            self.talk_button.configure(text="⏹ 结束并发送（空格键）")
            self.status_var.set("正在听你说英语……说完后再按一次空格键。")
        except Exception as exc:
            messagebox.showerror("麦克风无法使用", str(exc))

    def finish_recording(self) -> None:
        try:
            audio_path = self.recorder.stop()
        except Exception as exc:
            self.talk_button.configure(text="🎙 开始说英语（空格键）")
            messagebox.showerror("录音失败", str(exc))
            return
        self.talk_button.configure(text="🎙 开始说英语（空格键）")
        if not audio_path:
            self.status_var.set("录音太短，请重新说一次。")
            return
        self.busy = True
        self.stop_event = threading.Event()
        self.tts_queue = queue.Queue()
        self.tts_worker_event = None
        self.status_var.set("正在识别你的英语……")
        threading.Thread(target=self._conversation_worker, args=(audio_path,), daemon=True).start()

    def _conversation_worker(self, audio_path: Path) -> None:
        try:
            if self.whisper is None:
                raise RuntimeError("Whisper 模型还没有准备好，请稍等片刻。")
            segments, _ = self.whisper.transcribe(
                str(audio_path),
                language="en",
                beam_size=1,
                best_of=1,
                temperature=0.0,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300, "speech_pad_ms": 100},
                without_timestamps=True,
                condition_on_previous_text=False,
            )
            user_en = " ".join(segment.text.strip() for segment in segments).strip()
            if not user_en:
                raise RuntimeError("没有识别到语音。请靠近麦克风并说得稍长一些。")

            self.events.call(self._start_turn_display, user_en)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, *self.history[-12:], {"role": "user", "content": user_en}]
            raw = ""
            spoken_chars = 0
            for token in self.llm.stream(messages, self.stop_event):
                raw += token
                reply_en = tag_value(raw, "reply_en")
                user_zh = tag_value(raw, "user_zh")
                reply_zh = tag_value(raw, "reply_zh")
                self.events.call(self._update_turn_display, user_en, user_zh, reply_en, reply_zh)
                chunks, spoken_chars = complete_sentences(reply_en, spoken_chars)
                for chunk in chunks:
                    self._enqueue_tts(chunk)

            if self.stop_event.is_set():
                return
            reply_en = tag_value(raw, "reply_en", allow_partial=False)
            user_zh = tag_value(raw, "user_zh", allow_partial=False)
            reply_zh = tag_value(raw, "reply_zh", allow_partial=False)
            if not reply_en:
                reply_en = re.sub(r"<[^>]+>", "", raw).strip()
            chunks, spoken_chars = complete_sentences(reply_en, spoken_chars, force=True)
            for chunk in chunks:
                self._enqueue_tts(chunk)
            self.history.extend(({"role": "user", "content": user_en}, {"role": "assistant", "content": reply_en}))
            self.events.call(self._update_turn_display, user_en, user_zh, reply_en, reply_zh)
        except Exception as exc:
            self.events.call(self._show_error, str(exc))
        finally:
            # Always wake a TTS worker that may be waiting for the next sentence.
            self.tts_queue.put(None)
            try:
                audio_path.unlink(missing_ok=True)
            except OSError:
                pass
            self.events.call(self._processing_finished)

    def _enqueue_tts(self, text: str) -> None:
        self.tts_queue.put(text)
        if self.tts_worker_event is not self.stop_event:
            turn_stop_event = self.stop_event
            turn_queue = self.tts_queue
            self.tts_worker_event = turn_stop_event
            threading.Thread(target=self._tts_worker, args=(turn_stop_event, turn_queue), daemon=True).start()

    def _tts_worker(self, turn_stop_event: threading.Event, turn_queue: queue.Queue[str | None]) -> None:
        self.events.call(self.teacher.show_speaking)
        self.events.call(self.status_var.set, "老师正在说话……")
        try:
            if TTS_PROVIDER == "pyttsx3":
                engine = pyttsx3.init()
                engine.setProperty("rate", 155)
                while not turn_stop_event.is_set():
                    item = turn_queue.get()
                    if item is None:
                        break
                    engine.say(item)
                    engine.runAndWait()
                engine.stop()
            else:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                while not turn_stop_event.is_set():
                    item = turn_queue.get()
                    if item is None:
                        break
                    path = Path(tempfile.mkstemp(prefix="voice_tutor_tts_", suffix=".mp3")[1])
                    try:
                        asyncio.run(edge_tts.Communicate(text=item, voice=EDGE_TTS_VOICE).save(str(path)))
                        pygame.mixer.music.load(str(path))
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy() and not turn_stop_event.is_set():
                            time.sleep(0.03)
                        pygame.mixer.music.unload()
                    finally:
                        path.unlink(missing_ok=True)
        except Exception as exc:
            self.events.call(self.status_var.set, f"语音播放失败：{exc}")
        finally:
            self.events.call(self.teacher.show_idle)
            if turn_stop_event is self.stop_event:
                self.tts_worker_event = None
                self.events.call(self._set_not_busy)
            if not turn_stop_event.is_set() and turn_stop_event is self.stop_event:
                self.events.call(self.status_var.set, "轮到你了。点击按钮或按空格键开始说英语。")

    def _start_turn_display(self, user_en: str) -> None:
        self.status_var.set("本地模型正在回答……")
        self._append("\nYou\n", "user_head")
        self._append(user_en + "\n", "english")
        self._append("中文：翻译中……\n", "chinese")
        self._append("\nMila\n", "teacher_head")
        self._append("思考中……\n", "english")

    def _update_turn_display(self, user_en: str, user_zh: str, reply_en: str, reply_zh: str) -> None:
        # Repaint only the active turn. Earlier turns remain in self.history but the window
        # intentionally behaves like four large, readable live subtitle lines.
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.insert("end", "You\n", "user_head")
        self.transcript.insert("end", (user_en or "…") + "\n", "english")
        self.transcript.insert("end", "中文：" + (user_zh or "翻译中……") + "\n\n", "chinese")
        self.transcript.insert("end", "Mila\n", "teacher_head")
        self.transcript.insert("end", (reply_en or "思考中……") + "\n", "english")
        self.transcript.insert("end", "中文：" + (reply_zh or "翻译中……") + "\n", "chinese")
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def _append(self, text: str, tag: str) -> None:
        self.transcript.configure(state="normal")
        self.transcript.insert("end", text, tag)
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def _processing_finished(self) -> None:
        if self.tts_worker_event is not self.stop_event:
            self.busy = False
        if self.tts_worker_event is not self.stop_event and not self.stop_event.is_set():
            self.status_var.set("轮到你了。点击按钮或按空格键开始说英语。")

    def _set_not_busy(self) -> None:
        self.busy = False

    def _show_error(self, message: str) -> None:
        self.busy = False
        self.teacher.show_idle()
        self.status_var.set("发生错误，请检查 LM Studio、麦克风或网络。")
        messagebox.showerror("Mila", message)

    def stop_current(self) -> None:
        self.stop_event.set()
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
        while True:
            try:
                self.tts_queue.get_nowait()
            except queue.Empty:
                break
        self.tts_queue.put(None)
        self.tts_worker_event = None
        self.busy = False
        self.teacher.show_idle()
        self.status_var.set("已停止。可以开始下一轮。")

    def clear_history(self) -> None:
        self.stop_current()
        self.history.clear()
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")
        self.status_var.set("对话已清空。点击按钮或按空格键开始。")

    def close(self) -> None:
        self.stop_current()
        if self.recorder.recording:
            self.recorder.stop()
        self.root.destroy()


def main() -> None:
    missing = [path.name for path in (IDLE_GIF, SPEAKING_GIF) if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少老师图片：" + ", ".join(missing))
    root = tk.Tk()
    VoiceTutorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
