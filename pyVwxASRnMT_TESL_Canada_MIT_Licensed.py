# MIT License
#
# Copyright (c) 2026 Vince Wang and Daniel Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Authors: Vince Wang, Daniel Wang

import json
import math
import os
import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from rust_asio_portable import RustAsioBridge, ThreadedAsioReader

import numpy as np

try:
    import soxr
    HAVE_SOXR = True
except Exception:
    soxr = None
    HAVE_SOXR = False

try:
    from scipy.signal import resample_poly
    HAVE_SCIPY_RESAMPLE = True
except Exception:
    resample_poly = None
    HAVE_SCIPY_RESAMPLE = False
import requests
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, messagebox
from faster_whisper import WhisperModel

try:
    import pynvml
    HAVE_PYNVML = True
except Exception:
    pynvml = None
    HAVE_PYNVML = False

try:
    import pyaudiowpatch as pyaudio
    HAVE_PYAUDIOWPATCH = True
except Exception:
    pyaudio = None
    HAVE_PYAUDIOWPATCH = False

# ============================================================
# CONFIG
# ============================================================

WINDOW_W = 1920
WINDOW_H = 1080

DEFAULT_RUNTIME = "Auto"         # Auto / CPU / CUDA
DEFAULT_MODEL_PRESET = "turbo (large-v3 turbo) | compute=int8_float16 | UI name=q5"
DEFAULT_CHUNK_SECONDS = 7.0

ASR_SAMPLE_RATE = 16000
MIN_CHUNK_SECONDS = 1.5
VAD_FILTER = False

LLAMA_URL = "http://127.0.0.1:8033/v1/chat/completions"
LLAMA_MODEL = "Qwen3.5-9B"

LANG_CHOICES = [
    ("Auto", "auto"),
    ("English", "en"),
    ("Chinese", "zh"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("French", "fr"),
    ("German", "de"),
    ("Spanish", "es"),
    ("Russian", "ru"),
    ("Malay", "ms"),
]

RUNTIME_CHOICES = ["Auto", "CPU", "CUDA"]

# Explicit UI labels -> (faster-whisper model name, compute type)
MODEL_PRESETS = [
    ("medium | compute=int8 | UI name=q8", "medium", "int8"),
    ("medium | compute=int8_float16 | UI name=q5", "medium", "int8_float16"),
    ("large-v2 | compute=int8 | UI name=q8", "large-v2", "int8"),
    ("large-v2 | compute=int8_float16 | UI name=q5", "large-v2", "int8_float16"),
    ("turbo (large-v3 turbo) | compute=int8_float16 | UI name=q5", "turbo", "int8_float16"),
]

LEGACY_PRESET_ALIASES = {
    "medium q8": "medium | compute=int8 | UI name=q8",
    "medium q5": "medium | compute=int8_float16 | UI name=q5",
    "large-v2 q8": "large-v2 | compute=int8 | UI name=q8",
    "large-v2 q5": "large-v2 | compute=int8_float16 | UI name=q5",
    "large-v3 turbo q5": "turbo (large-v3 turbo) | compute=int8_float16 | UI name=q5",
}

TEXT_MAX_LINES = 1500

MODEL_VRAM_HINT_GB = {
    "medium | compute=int8_float16 | UI name=q5": 2.5,
    "medium | compute=int8 | UI name=q8": 3.2,
    "turbo (large-v3 turbo) | compute=int8_float16 | UI name=q5": 5.2,
    "large-v2 | compute=int8_float16 | UI name=q5": 6.2,
    "large-v2 | compute=int8 | UI name=q8": 7.8,
}

MODEL_QUALITY_ORDER = [
    "large-v2 | compute=int8 | UI name=q8",
    "large-v2 | compute=int8_float16 | UI name=q5",
    "turbo (large-v3 turbo) | compute=int8_float16 | UI name=q5",
    "medium | compute=int8 | UI name=q8",
    "medium | compute=int8_float16 | UI name=q5",
]

FALLBACK_ASIO_BUFFER_FRAMES = [64, 128, 256, 512, 1024, 2048, 4096]


# ============================================================
# HELPERS
# ============================================================

def safe_json_extract(text: str) -> str:
    if not text:
        return ""

    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "translation" in obj:
            return str(obj["translation"]).replace("\r\n", "\n").strip()
    except Exception:
        pass

    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        candidate = m.group(0)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "translation" in obj:
                return str(obj["translation"]).replace("\r\n", "\n").strip()
        except Exception:
            pass

    m = re.search(r'"translation"\s*:\s*"(.*)"', s, flags=re.DOTALL)
    if m:
        raw = m.group(1)
        try:
            decoded = json.loads(f'"{raw}"')
            return decoded.replace("\r\n", "\n").strip()
        except Exception:
            return raw.replace("\\n\\n", "\n\n").replace("\\n", "\n").strip()

    return s.replace("\r\n", "\n").strip()


def language_name_to_code(name: str) -> str:
    for label, code in LANG_CHOICES:
        if label == name:
            return code
    return "auto"


def code_to_language_name(code: str) -> str:
    for label, c in LANG_CHOICES:
        if c == code:
            return label
    return code


@dataclass
class AudioSourceChoice:
    label: str
    device_index: int
    kind: str          # "mic" or "loopback"
    backend: str       # "sounddevice" or "pyaudiowpatch" or "rust_asio"
    samplerate: int
    channels: int
    hostapi_name: str = ""
    device_id: str = ""
    raw: Optional[dict] = None


# ============================================================
# APP
# ============================================================

class FasterWhisperTkApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Faster-Whisper + Local MT")
        self.root.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.root.minsize(1200, 700)

        self.model: Optional[WhisperModel] = None
        self.model_loaded = False
        self.running = False
        self.start_in_progress = False
        self.start_cancelled = False
        self.start_thread: Optional[threading.Thread] = None
        self.stream = None
        self.pa = None
        self.pa_stream = None
        self.rust_asio = RustAsioBridge()
        self.rust_reader: Optional[ThreadedAsioReader] = None
        self.current_asio_runtime = {}

        self.audio_queue = queue.Queue(maxsize=512)
        self.ui_queue = queue.Queue()
        self.current_audio_chunks: List[np.ndarray] = []
        self.current_chunk_started_at = None
        self.capture_samplerate = ASR_SAMPLE_RATE

        self.available_sources: List[AudioSourceChoice] = []
        self.worker_thread = None
        self.translator_session = requests.Session()

        self.state_lock = threading.Lock()
        self.audio_peak = 0.0
        self.last_callback_debug_ts = 0.0
        self.debug_enabled = True

        self.orig_lang_var = tk.StringVar(value="Auto")
        self.dest_lang_var = tk.StringVar(value="English")
        self.auto_reverse_var = tk.BooleanVar(value=True)
        self.model_preset_var = tk.StringVar(value=DEFAULT_MODEL_PRESET)
        self.runtime_var = tk.StringVar(value=DEFAULT_RUNTIME)
        self.capture_mode_var = tk.StringVar(value="Microphone")
        self.source_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Idle")
        self.startup_status_var = tk.StringVar(value="Startup: idle")
        self.debug_var = tk.BooleanVar(value=True)
        self.meter_text_var = tk.StringVar(value="Peak: 0.0000")
        self.chunk_seconds_var = tk.DoubleVar(value=DEFAULT_CHUNK_SECONDS)
        self.auto_model_var = tk.BooleanVar(value=True)
        self.gpu_name_var = tk.StringVar(value="GPU: n/a")
        self.gpu_mem_var = tk.StringVar(value="GPU Mem: n/a")
        self.gpu_speed_var = tk.StringVar(value="ASR Speed: n/a")
        self.model_actual_var = tk.StringVar(value="Resolved model: not loaded")
        self.asio_channel_var = tk.StringVar(value="Auto")
        self.asio_buffer_choice_var = tk.StringVar(value="Auto")
        self.asio_buffer_label_var = tk.StringVar(value="Auto")
        self.asio_buffer_options: List[int] = []
        self.asio_channel_hint_var = tk.StringVar(value="ASIO Channels: Auto")
        self.asio_runtime_var = tk.StringVar(value="ASIO Runtime: n/a")
        self.mono_source_var = tk.StringVar(value="Auto")
        self.last_asr_metrics: Optional[dict] = None
        self.last_model_load_note = ""
        self.nvml_initialized = False
        self.nvml_device_index = 0

        self._build_ui()
        self._bind_vars()
        self._refresh_control_states()
        self._refresh_audio_sources()
        self._update_model_resolution_label()
        self._schedule_ui_pump()
        self._schedule_meter_update()

    # --------------------------------------------------------
    # VAR BINDING
    # --------------------------------------------------------

    def _bind_vars(self):
        self.debug_enabled = bool(self.debug_var.get())
        self.debug_var.trace_add("write", self._on_debug_toggle)
        self.chunk_seconds_var.trace_add("write", self._on_chunk_slider_changed)
        self.asio_buffer_choice_var.trace_add("write", self._on_asio_buffer_changed)
        self.capture_mode_var.trace_add("write", self._on_capture_mode_changed)
        self.source_var.trace_add("write", self._on_source_changed)
        self.asio_channel_var.trace_add("write", self._on_asio_channel_changed)
        self.mono_source_var.trace_add("write", self._on_mono_source_changed)
        self.model_preset_var.trace_add("write", self._update_model_resolution_label)
        self.runtime_var.trace_add("write", self._update_model_resolution_label)
        self.auto_model_var.trace_add("write", self._update_model_resolution_label)

    def _on_debug_toggle(self, *_args):
        self.debug_enabled = bool(self.debug_var.get())

    def _on_chunk_slider_changed(self, *_args):
        if hasattr(self, "chunk_label_var"):
            self.chunk_label_var.set(f"{self.chunk_seconds_var.get():.1f} sec")

    def _on_asio_buffer_changed(self, *_args):
        self._update_asio_buffer_hint()

    def _on_capture_mode_changed(self, *_args):
        self._update_asio_controls_visibility()

    def _on_source_changed(self, *_args):
        self._populate_asio_channel_options()
        self._populate_asio_buffer_options()
        self._populate_mono_source_options()

    def _on_asio_channel_changed(self, *_args):
        self._update_asio_channel_hint()

    def _on_mono_source_changed(self, *_args):
        self._update_mono_source_hint()



    def _get_mono_source_options(self, source: Optional[AudioSourceChoice] = None) -> List[str]:
        if source is None:
            source = self._get_selected_source()
        max_channels = max(1, int(source.channels if source else 1))
        return ["Auto"] + [f"CH{i}" for i in range(max_channels)]

    def _populate_mono_source_options(self):
        if not hasattr(self, "mono_source_combo"):
            return
        source = self._get_selected_source()
        values = self._get_mono_source_options(source)
        self.mono_source_combo["values"] = values
        current = (self.mono_source_var.get() or "").strip()
        if current not in values:
            self.mono_source_var.set("Auto")
        self._update_mono_source_hint()

    def _forced_mono_channel_index(self) -> Optional[int]:
        s = (self.mono_source_var.get() or "").strip().upper()
        if not s or s == "AUTO":
            return None
        if s.startswith("CH"):
            try:
                return max(0, int(s[2:]))
            except Exception:
                return None
        return None

    def _update_mono_source_hint(self):
        if not hasattr(self, "mono_source_hint_var"):
            return
        forced = self._forced_mono_channel_index()
        if forced is None:
            self.mono_source_hint_var.set("ASR Mono: Auto = strongest single channel, no phase-mixing")
        else:
            self.mono_source_hint_var.set(f"ASR Mono: forced {self.mono_source_var.get()}")

    def _refresh_control_states(self):
        if hasattr(self, "start_button"):
            self.start_button.state(["disabled"] if (self.start_in_progress or self.running) else ["!disabled"])
        if hasattr(self, "stop_button"):
            self.stop_button.state(["!disabled"] if (self.start_in_progress or self.running) else ["disabled"])

    def _set_startup_status(self, text: str):
        self.startup_status_var.set(text)
        self._refresh_control_states()

    def _update_asio_controls_visibility(self):
        if not hasattr(self, "asio_controls_frame"):
            return
        if self.capture_mode_var.get() == "ASIO (Rust FFI)":
            self.asio_controls_frame.pack(fill=tk.X, pady=(0, 4))
        else:
            self.asio_controls_frame.pack_forget()

    def _extract_asio_buffer_caps(self, source: Optional[AudioSourceChoice]) -> tuple[int, int, int, int]:
        raw = (source.raw or {}) if source else {}
        min_frames = int(raw.get("min_buffer_frames", 0) or 0)
        preferred_frames = int(raw.get("preferred_buffer_frames", 0) or 0)
        max_frames = int(raw.get("max_buffer_frames", 0) or 0)
        granularity = int(raw.get("buffer_granularity", raw.get("granularity", 0)) or 0)
        return min_frames, preferred_frames, max_frames, granularity

    def _enumerate_asio_buffer_frames(self, source: Optional[AudioSourceChoice]) -> List[int]:
        min_frames, preferred_frames, max_frames, granularity = self._extract_asio_buffer_caps(source)
        values = set()

        if min_frames > 0:
            values.add(min_frames)
        if preferred_frames > 0:
            values.add(preferred_frames)
        if max_frames > 0:
            values.add(max_frames)

        if min_frames > 0 and max_frames >= min_frames:
            if granularity == 0:
                if preferred_frames > 0:
                    values.add(preferred_frames)
            elif granularity == -1:
                v = max(1, min_frames)
                while v <= max_frames:
                    values.add(v)
                    nv = v * 2
                    if nv <= v:
                        break
                    v = nv
            else:
                step = max(1, granularity)
                count = ((max_frames - min_frames) // step) + 1
                if count <= 256:
                    for v in range(min_frames, max_frames + 1, step):
                        values.add(v)
                else:
                    for ratio in [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]:
                        v = min_frames + int(round((max_frames - min_frames) * ratio / step)) * step
                        v = max(min_frames, min(max_frames, v))
                        values.add(v)

        if not values:
            values.update(FALLBACK_ASIO_BUFFER_FRAMES)

        return sorted(v for v in values if v > 0)

    def _get_selected_asio_buffer_frames(self, source: Optional[AudioSourceChoice] = None) -> int:
        if source is None:
            source = self._get_selected_source()
        options = self._enumerate_asio_buffer_frames(source)
        choice = (self.asio_buffer_choice_var.get() or "Auto").strip()
        if choice and choice.lower() != "auto":
            m = re.match(r"^(\d+)", choice)
            if m:
                wanted = int(m.group(1))
                if wanted in options:
                    return wanted
                return min(options, key=lambda x: abs(x - wanted))
        _, preferred_frames, _, _ = self._extract_asio_buffer_caps(source)
        if preferred_frames > 0:
            return preferred_frames
        return options[0] if options else FALLBACK_ASIO_BUFFER_FRAMES[0]

    def _populate_asio_buffer_options(self):
        if not hasattr(self, "asio_buffer_combo"):
            return
        source = self._get_selected_source()
        if self.capture_mode_var.get() != "ASIO (Rust FFI)" or not source or source.backend != "rust_asio":
            self.asio_buffer_options = []
            self.asio_buffer_combo["values"] = ["Auto"]
            self.asio_buffer_choice_var.set("Auto")
            self._update_asio_buffer_hint()
            return

        self.asio_buffer_options = self._enumerate_asio_buffer_frames(source)
        sr = max(1, int(source.samplerate or 48000))
        values = ["Auto"] + [f"{frames} frames ({self._format_buffer_ms_from_frames(frames, sr)})" for frames in self.asio_buffer_options]
        self.asio_buffer_combo["values"] = values

        current = (self.asio_buffer_choice_var.get() or "").strip()
        current_frames = None
        m = re.match(r"^(\d+)", current)
        if m:
            current_frames = int(m.group(1))
        preferred = self._get_selected_asio_buffer_frames(source)
        if current_frames in self.asio_buffer_options:
            self.asio_buffer_choice_var.set(f"{current_frames} frames ({self._format_buffer_ms_from_frames(current_frames, sr)})")
        else:
            self.asio_buffer_choice_var.set(f"{preferred} frames ({self._format_buffer_ms_from_frames(preferred, sr)})")
        self._update_asio_buffer_hint()

    def _update_asio_buffer_hint(self):
        if not hasattr(self, "asio_buffer_label_var"):
            return
        source = self._get_selected_source()
        if self.capture_mode_var.get() != "ASIO (Rust FFI)" or not source or source.backend != "rust_asio":
            self.asio_buffer_label_var.set("Auto")
            return
        frames = self._get_selected_asio_buffer_frames(source)
        min_frames, preferred_frames, max_frames, granularity = self._extract_asio_buffer_caps(source)
        sr = max(1, int(source.samplerate or 48000))
        gran_text = "driver discrete" if granularity == -1 else ("preferred-only" if granularity == 0 else f"step {granularity}")
        pref_text = f", preferred {preferred_frames}" if preferred_frames > 0 else ""
        cap_text = f"{min_frames}-{max_frames} frames" if min_frames > 0 and max_frames >= min_frames else "driver caps n/a"
        self.asio_buffer_label_var.set(
            f"{frames} frames ({self._format_buffer_ms_from_frames(frames, sr)}), caps {cap_text}{pref_text}, {gran_text}"
        )

    def _format_buffer_ms_from_frames(self, frames: int, sample_rate: int) -> str:
        sr = max(1, int(sample_rate or 48000))
        return f"{(1000.0 * frames / sr):.2f} ms"

    def _get_selected_asio_channels(self):
        s = self.asio_channel_var.get().strip()
        if not s or s.lower() == "auto":
            return []
        s = s.upper().replace(" ", "")
        if s.startswith("CH"):
            parts = s.split("+")
            out = []
            for p in parts:
                if p.startswith("CH"):
                    p = p[2:]
                if p:
                    out.append(int(p))
            return sorted(dict.fromkeys(out))
        raw_parts = s.replace("+", ",").split(",")
        out = []
        for p in raw_parts:
            p = p.strip()
            if not p:
                continue
            out.append(int(p))
        return sorted(dict.fromkeys(out))

    def _get_effective_asio_channel_count(self, source=None) -> int:
        selected = self._get_selected_asio_channels()
        if selected:
            return len(selected)
        if source is None:
            source = self._get_selected_source()
        return max(1, int(source.channels if source else 1))

    def _make_asio_channel_presets(self, max_channels: int):
        max_channels = max(1, int(max_channels))
        vals = ["Auto"]
        vals.extend([f"CH{i}" for i in range(max_channels)])
        for i in range(max_channels - 1):
            vals.append(f"CH{i}+CH{i+1}")
        for i in range(1, max_channels):
            combo = f"CH0+CH{i}"
            if combo not in vals:
                vals.append(combo)
        if max_channels >= 4:
            vals.append("CH0+CH1+CH2+CH3")
        return vals

    def _populate_asio_channel_options(self):
        if not hasattr(self, "asio_channel_combo"):
            return
        source = self._get_selected_source()
        if self.capture_mode_var.get() != "ASIO (Rust FFI)" or not source or source.backend != "rust_asio":
            self.asio_channel_combo["values"] = ["Auto"]
            self.asio_channel_var.set("Auto")
            self._update_asio_channel_hint()
            return
        presets = self._make_asio_channel_presets(source.channels)
        self.asio_channel_combo["values"] = presets
        if not self.asio_channel_var.get().strip():
            self.asio_channel_var.set("Auto")
        self._update_asio_channel_hint()

    def _update_asio_channel_hint(self):
        if not hasattr(self, "asio_channel_hint_var"):
            return
        source = self._get_selected_source()
        if self.capture_mode_var.get() != "ASIO (Rust FFI)" or not source or source.backend != "rust_asio":
            self.asio_channel_hint_var.set("ASIO Channels: n/a")
            return
        selected = self._get_selected_asio_channels()
        if selected:
            self.asio_channel_hint_var.set(
                f"ASIO Channels: requested {selected} -> capture {len(selected)} ch"
            )
        else:
            self.asio_channel_hint_var.set(
                f"ASIO Channels: Auto -> capture {max(1, int(source.channels))} ch"
            )

    # --------------------------------------------------------
    # MODEL PRESETS
    # --------------------------------------------------------

    def _preset_labels(self):
        return [x[0] for x in MODEL_PRESETS]

    def _resolve_model_preset(self):
        selected = LEGACY_PRESET_ALIASES.get(self.model_preset_var.get().strip(), self.model_preset_var.get().strip())
        for label, model_name, compute_type in MODEL_PRESETS:
            if label == selected:
                return model_name, compute_type
        return "turbo", "int8_float16"

    def _normalize_preset_label(self, label: str) -> str:
        return LEGACY_PRESET_ALIASES.get(label.strip(), label.strip())

    def _format_resolved_model_text(self, selected_label: str, selection_note: str) -> str:
        model_name, compute_type = self._resolve_model_preset_from_label(selected_label)
        runtime_choice = self.runtime_var.get().strip().lower()
        auto_note = "auto-preset active" if self.auto_model_var.get() and runtime_choice != "cpu" else "manual preset"
        return f"Resolved model: {selected_label} -> model={model_name}, compute={compute_type} [{selection_note}; {auto_note}]"

    def _resolve_model_preset_from_label(self, label: str):
        normalized = self._normalize_preset_label(label)
        for preset_label, model_name, compute_type in MODEL_PRESETS:
            if preset_label == normalized:
                return model_name, compute_type
        return "turbo", "int8_float16"

    def _update_model_resolution_label(self, *_args):
        selected_label, selection_note = self._choose_vram_aware_preset()
        self.model_actual_var.set(self._format_resolved_model_text(selected_label, selection_note))

    def _init_nvml(self):
        if not HAVE_PYNVML:
            return False
        if self.nvml_initialized:
            return True
        try:
            pynvml.nvmlInit()
            self.nvml_initialized = True
            return True
        except Exception as e:
            self.debug(f"NVML init failed: {e}", to_ui=False)
            return False

    def _get_gpu_stats(self) -> Optional[dict]:
        if not self._init_nvml():
            return None
        try:
            count = pynvml.nvmlDeviceGetCount()
            if count < 1:
                return None
            index = min(max(self.nvml_device_index, 0), count - 1)
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            try:
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
            except Exception:
                name = f"GPU {index}"
            return {
                "index": index,
                "name": str(name),
                "total_gb": mem.total / (1024 ** 3),
                "used_gb": mem.used / (1024 ** 3),
                "free_gb": mem.free / (1024 ** 3),
                "gpu_util": float(getattr(util, "gpu", 0.0)),
                "mem_util": float(getattr(util, "memory", 0.0)),
            }
        except Exception as e:
            self.debug(f"NVML read failed: {e}", to_ui=False)
            return None

    def _choose_vram_aware_preset(self) -> Tuple[str, str]:
        selected = self._normalize_preset_label(self.model_preset_var.get().strip())
        runtime_choice = self.runtime_var.get().strip().lower()
        if not self.auto_model_var.get() or runtime_choice == "cpu":
            return selected, "manual"

        stats = self._get_gpu_stats()
        if not stats:
            return selected, "manual (no_gpu_stats)"

        free_gb = stats["free_gb"]
        need = MODEL_VRAM_HINT_GB.get(selected, None)
        if need is None:
            return selected, f"manual (free={free_gb:.1f}GB, no_hint)"

        cushion_gb = 1.2
        if free_gb >= need + cushion_gb:
            return selected, f"manual OK (free={free_gb:.1f}GB >= need~{need + cushion_gb:.1f}GB)"
        return selected, f"manual forced (free={free_gb:.1f}GB < suggested~{need + cushion_gb:.1f}GB)"

    # --------------------------------------------------------
    # LOGGING / DEBUG
    # --------------------------------------------------------

    def debug(self, msg: str, to_ui: bool = True):
        ts = time.strftime("%H:%M:%S")
        line = f"[Dbg {ts}] {msg}"
        print(line, flush=True)

        if self.debug_enabled and to_ui:
            try:
                self.ui_queue.put_nowait(("append_debug", line + "\n"))
            except queue.Full:
                pass

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        row1 = ttk.Frame(top)
        row1.pack(fill=tk.X, pady=4)

        ttk.Label(row1, text="Orig-Lang:").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Combobox(
            row1,
            textvariable=self.orig_lang_var,
            values=[x[0] for x in LANG_CHOICES],
            state="readonly",
            width=14,
        ).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(row1, text="Dest-Lang:").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Combobox(
            row1,
            textvariable=self.dest_lang_var,
            values=[x[0] for x in LANG_CHOICES if x[0] != "Auto"],
            state="readonly",
            width=14,
        ).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Checkbutton(
            row1,
            text="Auto reverse direction",
            variable=self.auto_reverse_var
        ).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Button(row1, text="Swap", command=self.swap_languages).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(row1, text="Capture:").pack(side=tk.LEFT, padx=(20, 6))
        cap_combo = ttk.Combobox(
            row1,
            textvariable=self.capture_mode_var,
            values=["Microphone", "WASAPI Loopback", "ASIO (Rust FFI)"],
            state="readonly",
            width=18,
        )
        cap_combo.pack(side=tk.LEFT, padx=(0, 8))
        cap_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_audio_sources())

        ttk.Button(row1, text="Refresh Devices", command=self._refresh_audio_sources).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(row1, text="Debug", variable=self.debug_var).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Checkbutton(row1, text="VRAM auto model", variable=self.auto_model_var).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(row1, text="Chunk:").pack(side=tk.LEFT, padx=(8, 6))
        chunk_scale = ttk.Scale(
            row1,
            from_=1.0,
            to=30.0,
            variable=self.chunk_seconds_var,
            orient=tk.HORIZONTAL,
            length=180,
        )
        chunk_scale.pack(side=tk.LEFT, padx=(0, 8))

        self.chunk_label_var = tk.StringVar(value=f"{self.chunk_seconds_var.get():.1f} sec")
        ttk.Label(row1, textvariable=self.chunk_label_var, width=8).pack(side=tk.LEFT)

        ttk.Label(row1, text="ASR Mono:").pack(side=tk.LEFT, padx=(16, 6))
        self.mono_source_combo = ttk.Combobox(
            row1,
            textvariable=self.mono_source_var,
            values=["Auto", "CH0", "CH1"],
            state="readonly",
            width=8,
        )
        self.mono_source_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.mono_source_hint_var = tk.StringVar(value="ASR Mono: Auto = strongest single channel, no phase-mixing")
        ttk.Label(row1, textvariable=self.mono_source_hint_var).pack(side=tk.LEFT, padx=(0, 8))

        row2 = ttk.Frame(top)
        row2.pack(fill=tk.X, pady=4)

        ttk.Button(row2, text="Load Model", command=self.load_model).pack(side=tk.RIGHT, padx=(12, 0))

        ttk.Label(row2, text="Input Source:").pack(side=tk.LEFT, padx=(0, 6))
        self.source_combo = ttk.Combobox(
            row2,
            textvariable=self.source_var,
            state="readonly",
            width=60,
        )
        self.source_combo.pack(side=tk.LEFT, padx=(0, 12), fill=tk.X, expand=True)

        ttk.Label(row2, text="ASR Model:").pack(side=tk.LEFT, padx=(12, 6))
        ttk.Combobox(
            row2,
            textvariable=self.model_preset_var,
            values=self._preset_labels(),
            state="readonly",
            width=52,
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(row2, text="Runtime:").pack(side=tk.LEFT, padx=(12, 6))
        ttk.Combobox(
            row2,
            textvariable=self.runtime_var,
            values=RUNTIME_CHOICES,
            state="readonly",
            width=8,
        ).pack(side=tk.LEFT, padx=(0, 8))

        row2b = ttk.Frame(top)
        row2b.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(row2b, textvariable=self.model_actual_var).pack(side=tk.LEFT, padx=(0, 6))

        self.asio_controls_frame = ttk.Frame(top)

        ttk.Label(self.asio_controls_frame, text="ASIO Channels:").pack(side=tk.LEFT, padx=(0, 6))
        self.asio_channel_combo = ttk.Combobox(
            self.asio_controls_frame,
            textvariable=self.asio_channel_var,
            values=["Auto"],
            state="normal",
            width=24,
        )
        self.asio_channel_combo.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(self.asio_controls_frame, text="ASIO Buffer:").pack(side=tk.LEFT, padx=(0, 6))
        self.asio_buffer_combo = ttk.Combobox(
            self.asio_controls_frame,
            textvariable=self.asio_buffer_choice_var,
            values=["Auto"],
            state="readonly",
            width=26,
        )
        self.asio_buffer_combo.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(self.asio_controls_frame, textvariable=self.asio_buffer_label_var, width=48).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(self.asio_controls_frame, textvariable=self.asio_channel_hint_var).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(self.asio_controls_frame, textvariable=self.asio_runtime_var).pack(side=tk.LEFT, padx=(0, 6))

        row3 = ttk.Frame(top)
        row3.pack(fill=tk.X, pady=4)

        self.start_button = ttk.Button(row3, text="Start", command=self.start_capture)
        self.start_button.pack(side=tk.LEFT, padx=(0, 8))
        self.stop_button = ttk.Button(row3, text="Stop", command=self.stop_capture)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(row3, text="Clear ASR", command=self.clear_asr_text).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(row3, text="Clear MT", command=self.clear_mt_text).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(row3, text="Dump Devices", command=self.dump_devices_debug).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(row3, textvariable=self.status_var).pack(side=tk.RIGHT)

        row3b = ttk.Frame(top)
        row3b.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(row3b, textvariable=self.startup_status_var).pack(side=tk.LEFT)

        row4 = ttk.Frame(top)
        row4.pack(fill=tk.X, pady=(4, 4))

        ttk.Label(row4, text="Audio / Mic Level:").pack(side=tk.LEFT, padx=(0, 8))

        self.meter_canvas = tk.Canvas(row4, width=420, height=20, highlightthickness=1, highlightbackground="#777")
        self.meter_canvas.pack(side=tk.LEFT)
        self.meter_canvas.create_rectangle(0, 0, 420, 20, fill="#202020", outline="")
        self.meter_bar = self.meter_canvas.create_rectangle(0, 0, 0, 20, fill="#30c060", outline="")

        ttk.Label(row4, textvariable=self.meter_text_var).pack(side=tk.LEFT, padx=(10, 16))
        ttk.Label(row4, textvariable=self.gpu_name_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(row4, textvariable=self.gpu_mem_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Label(row4, textvariable=self.gpu_speed_var).pack(side=tk.LEFT, padx=(0, 0))

        pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left_frame = ttk.Frame(pane)
        right_frame = ttk.Frame(pane)
        pane.add(left_frame, weight=1)
        pane.add(right_frame, weight=1)

        ttk.Label(left_frame, text="ASR (Mic / WASAPI Loopback)").pack(anchor="w")
        left_text_frame = ttk.Frame(left_frame)
        left_text_frame.pack(fill=tk.BOTH, expand=True)

        self.asr_text = tk.Text(
            left_text_frame,
            wrap="word",
            font=("Noto Sans", 11),
            spacing1=0,
            spacing2=0,
            spacing3=1,
            pady=2,
        )
        self.asr_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.asr_scrollbar = ttk.Scrollbar(left_text_frame, orient="vertical", command=self.asr_text.yview)
        self.asr_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.asr_text.configure(yscrollcommand=self.asr_scrollbar.set)

        ttk.Label(right_frame, text="Machine Translation").pack(anchor="w")
        right_text_frame = ttk.Frame(right_frame)
        right_text_frame.pack(fill=tk.BOTH, expand=True)

        self.mt_text = tk.Text(
            right_text_frame,
            wrap="word",
            font=("Consolas", 11),
            spacing1=0,
            spacing2=0,
            spacing3=1,
            pady=2,
        )
        self.mt_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.mt_scrollbar = ttk.Scrollbar(right_text_frame, orient="vertical", command=self.mt_text.yview)
        self.mt_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.mt_text.configure(yscrollcommand=self.mt_scrollbar.set)
        self._update_asio_controls_visibility()

    # --------------------------------------------------------
    # DEVICE ENUMERATION
    # --------------------------------------------------------

    def dump_devices_debug(self):
        try:
            self.debug(f"sounddevice version = {sd.__version__}")
            self.debug(f"PortAudio version = {sd.get_portaudio_version()}")
            hostapis = sd.query_hostapis()
            self.debug(f"Host APIs ({len(hostapis)}):")
            for i, ha in enumerate(hostapis):
                self.debug(f"  hostapi[{i}] = {ha}")

            devices = sd.query_devices()
            self.debug(f"sounddevice devices ({len(devices)}):")
            for i, d in enumerate(devices):
                host_name = hostapis[d["hostapi"]]["name"]
                self.debug(
                    f"  sd[{i}] host='{host_name}' name='{d['name']}' "
                    f"in={d['max_input_channels']} out={d['max_output_channels']} "
                    f"default_sr={d['default_samplerate']}"
                )

            if HAVE_PYAUDIOWPATCH:
                if self.pa is None:
                    self.pa = pyaudio.PyAudio()
                self.debug("PyAudioWPatch loopback devices:")
                for info in self.pa.get_loopback_device_info_generator():
                    self.debug(
                        f"  pa[{info.get('index')}] name='{info.get('name')}' "
                        f"in={info.get('maxInputChannels')} out={info.get('maxOutputChannels')} "
                        f"default_sr={info.get('defaultSampleRate')}"
                    )
            else:
                self.debug("PyAudioWPatch not installed")

            if self.rust_asio.available():
                self.debug(f"Rust portable ASIO wrapper loaded: {self.rust_asio.dll_path}")
                try:
                    self.debug("Rust ASIO devices:")
                    for dev in self.rust_asio.list_devices():
                        self.debug(
                            f"  asio id='{dev.id}' "
                            f"name='{dev.name}' "
                            f"channels={dev.channels} "
                            f"sample_rate={dev.sample_rate} "
                            f"preferred_buf={dev.preferred_buffer_frames}"
                        )
                except Exception as e:
                    self.debug(f"Rust ASIO enumeration failed: {e}")
            else:
                self.debug(f"Rust DLL unavailable: {self.rust_asio.load_error()}")
        except Exception as e:
            self.debug(f"dump_devices_debug failed: {e}")

    def _refresh_audio_sources(self):
        sources: List[AudioSourceChoice] = []
        capture_mode = self.capture_mode_var.get()
        want_loopback = capture_mode == "WASAPI Loopback"
        want_asio_only = capture_mode == "ASIO (Rust FFI)"
        want_microphones = capture_mode == "Microphone"

        if want_microphones:
            try:
                devices = sd.query_devices()
                hostapis = sd.query_hostapis()

                for i, d in enumerate(devices):
                    hostapi_name = hostapis[d["hostapi"]]["name"]
                    name = d["name"]
                    max_input = int(d["max_input_channels"])
                    default_sr = int(d["default_samplerate"]) if d["default_samplerate"] else 48000

                    if "WASAPI" in hostapi_name.upper() and max_input > 0:
                        sources.append(
                            AudioSourceChoice(
                                label=f"[Windows WASAPI][sounddevice] {name} | in={max_input} | {default_sr} Hz",
                                device_index=i,
                                kind="mic",
                                backend="sounddevice",
                                samplerate=default_sr,
                                channels=max_input,
                                hostapi_name=hostapi_name,
                            )
                        )

                for i, d in enumerate(devices):
                    hostapi_name = hostapis[d["hostapi"]]["name"]
                    name = d["name"]
                    max_input = int(d["max_input_channels"])
                    default_sr = int(d["default_samplerate"]) if d["default_samplerate"] else 48000

                    if max_input > 0 and "WASAPI" not in hostapi_name.upper():
                        sources.append(
                            AudioSourceChoice(
                                label=f"[{hostapi_name}][sounddevice] {name} | in={max_input} | {default_sr} Hz",
                                device_index=i,
                                kind="mic",
                                backend="sounddevice",
                                samplerate=default_sr,
                                channels=max_input,
                                hostapi_name=hostapi_name,
                            )
                        )
            except Exception as e:
                self.debug(f"sounddevice enumeration failed: {e}")

        if want_loopback:
            if not HAVE_PYAUDIOWPATCH:
                self.debug("PyAudioWPatch not installed; WASAPI loopback unavailable")
            else:
                try:
                    if self.pa is None:
                        self.pa = pyaudio.PyAudio()

                    for info in self.pa.get_loopback_device_info_generator():
                        max_input = int(info.get("maxInputChannels", 0))
                        default_sr = int(info.get("defaultSampleRate", 48000))
                        name = info.get("name", "Unknown loopback")
                        dev_index = int(info["index"])

                        if max_input > 0:
                            sources.append(
                                AudioSourceChoice(
                                    label=f"[WASAPI Loopback][PyAudioWPatch] {name} | in={max_input} | {default_sr} Hz",
                                    device_index=dev_index,
                                    kind="loopback",
                                    backend="pyaudiowpatch",
                                    samplerate=default_sr,
                                    channels=max_input,
                                    hostapi_name="WASAPI",
                                )
                            )
                except Exception as e:
                    self.debug(f"PyAudioWPatch loopback enumeration failed: {e}")

        if want_asio_only:
            if self.rust_asio.available():
                try:
                    for dev in self.rust_asio.list_devices():
                        dev_id = str(dev.id or dev.name or "").strip()
                        dev_name = str(dev.name or dev_id or "ASIO Device").strip()
                        if not dev_id:
                            continue
                        sr = int(dev.sample_rate or 48000)
                        ch = int(dev.channels or 2)
                        if ch < 1:
                            ch = 1
                        sources.append(
                            AudioSourceChoice(
                                label=f"[ASIO][Rust FFI] {dev_name} | in={ch} | {sr} Hz",
                                device_index=-1,
                                kind="mic",
                                backend="rust_asio",
                                samplerate=sr,
                                channels=ch,
                                hostapi_name="ASIO",
                                device_id=dev_id,
                                raw=dev.raw,
                            )
                        )
                except Exception as e:
                    self.debug(f"Rust ASIO device enumeration failed: {e}")
            else:
                self.debug(f"Rust DLL unavailable: {self.rust_asio.load_error()}")

        self.available_sources = sources
        labels = [s.label for s in sources]
        self.source_combo["values"] = labels
        self.source_var.set(labels[0] if labels else "")
        self._populate_asio_channel_options()
        self._populate_asio_buffer_options()
        self._populate_mono_source_options()

        self.debug(f"Refreshed sources for mode='{self.capture_mode_var.get()}': {len(sources)} source(s)")
        for s in sources[:50]:
            self.debug(
                f"  source: backend={s.backend} idx={s.device_index} kind={s.kind} "
                f"sr={s.samplerate} ch={s.channels} label={s.label}"
            )

    def _get_selected_source(self) -> Optional[AudioSourceChoice]:
        chosen = self.source_var.get()
        for s in self.available_sources:
            if s.label == chosen:
                return s
        return None

    def _normalize_device_name(self, label: str) -> str:
        s = label
        s = re.sub(r"^\[[^\]]+\]\[[^\]]+\]\s*", "", s)
        s = re.sub(r"^\[[^\]]+\]\s*", "", s)
        s = re.sub(r"\s+\|\s+in=.*$", "", s)
        s = re.sub(r"\s+\|\s+out=.*$", "", s)
        return s.strip().lower()

    def _find_fallback_non_wasapi_source(self, source: AudioSourceChoice) -> Optional[AudioSourceChoice]:
        target = self._normalize_device_name(source.label)
        for s in self.available_sources:
            if s is source:
                continue
            if s.kind != "mic":
                continue
            if s.backend != "sounddevice":
                continue
            if "WASAPI" in s.hostapi_name.upper():
                continue
            if self._normalize_device_name(s.label) == target:
                return s
        return None

    # --------------------------------------------------------
    # MODEL / RUNTIME
    # --------------------------------------------------------

    def _resolve_runtime(self):
        choice = self.runtime_var.get().strip().lower()
        selected_label, selection_note = self._choose_vram_aware_preset()

        model_name = "turbo"
        preset_compute = "int8_float16"
        for label, candidate_model_name, compute_type in MODEL_PRESETS:
            if label == selected_label:
                model_name = candidate_model_name
                preset_compute = compute_type
                break

        self.last_model_load_note = f"preset={selected_label} ({selection_note})"
        self.model_actual_var.set(self._format_resolved_model_text(selected_label, selection_note))

        if choice == "cpu":
            compute = "int8" if preset_compute in ("int8", "int8_float16") else "float32"
            return "cpu", model_name, compute, selected_label

        if choice == "cuda":
            return "cuda", model_name, preset_compute, selected_label

        stats = self._get_gpu_stats()
        if stats:
            return "cuda", model_name, preset_compute, selected_label

        compute = "int8" if preset_compute in ("int8", "int8_float16") else "float32"
        return "cpu", model_name, compute, selected_label

    def load_model(self):
        if self.model_loaded:
            self.status_var.set("Model already loaded")
            self.debug("Load Model skipped: model already loaded")
            return

        def _worker():
            try:
                device, model_name, compute, preset_label = self._resolve_runtime()
                runtime_choice = self.runtime_var.get().strip()

                self.ui_queue.put(("status", f"Loading {preset_label} ({device}/{compute})..."))
                self.debug(
                    f"Loading preset='{preset_label}', model='{model_name}', "
                    f"runtime_choice='{runtime_choice}', resolved_device='{device}', compute='{compute}', "
                    f"note='{self.last_model_load_note}'"
                )

                self.model = None
                self.model = WhisperModel(
                    model_name,
                    device=device,
                    compute_type=compute,
                )
                self.model_loaded = True

                self.ui_queue.put(("status", f"Model loaded: {preset_label} ({device}/{compute})"))
                self.model_actual_var.set(f"Loaded model: {preset_label} -> model={model_name}, compute={compute}, runtime={device}")
                self.debug("Whisper model loaded successfully")

            except Exception as e:
                self.model = None
                self.model_loaded = False
                self.ui_queue.put(("error", f"Failed to load model:\n{e}"))

        threading.Thread(target=_worker, daemon=True).start()

    def _translate_with_llama(self, text: str, orig_code: str, dest_code: str) -> str:
        system_prompt = (
            "You are a translation engine. "
            "Translate the user's text accurately and naturally. "
            "Preserve paragraph breaks. "
            "Return JSON only in this exact format: "
            '{"translation":"..."}'
        )

        if orig_code == "auto":
            user_prompt = (
                f"Translate the following text into {dest_code}. "
                f"Detect the source language automatically.\n\n{text}"
            )
        else:
            user_prompt = (
                f"Translate from {orig_code} to {dest_code}. "
                f"Preserve paragraph structure.\n\n{text}"
            )

        payload = {
            "model": LLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "stream": False,
        }

        self.debug(f"Sending MT request: {orig_code} -> {dest_code}, chars={len(text)}")
        r = self.translator_session.post(LLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        self.debug(f"MT raw response chars={len(content)}")
        return safe_json_extract(content)

    # --------------------------------------------------------
    # AUDIO BACKENDS
    # --------------------------------------------------------

    def _selected_source_channels(self) -> int:
        src = self._get_selected_source()
        return src.channels if src else 1

    def _capture_accepting_audio(self) -> bool:
        return bool(self.running or self.start_in_progress)

    def _safe_rust_asio_last_error(self) -> str:
        try:
            s = str(self.rust_asio.last_error() or "").strip()
            return s
        except Exception:
            return ""

    def _safe_rust_asio_runtime_snapshot(self) -> dict:
        try:
            return self.rust_asio.runtime_snapshot()
        except Exception:
            return {}

    def _cleanup_audio_backends(self):
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception as e:
            self.debug(f"sounddevice cleanup exception: {e}")
        self.stream = None

        try:
            if self.pa_stream is not None:
                self.pa_stream.stop_stream()
                self.pa_stream.close()
        except Exception as e:
            self.debug(f"PyAudioWPatch cleanup exception: {e}")
        self.pa_stream = None

        try:
            if self.rust_reader is not None:
                self.rust_reader.stop()
        except Exception as e:
            self.debug(f"Rust ASIO reader cleanup exception: {e}")
        self.rust_reader = None

        try:
            self.rust_asio.stop()
        except Exception as e:
            self.debug(f"Rust ASIO cleanup stop exception: {e}")

    def _open_sounddevice_input(self, source: AudioSourceChoice):
        devinfo = sd.query_devices(source.device_index)
        self.debug(f"Selected sounddevice info: {devinfo}")

        native_sr = int(devinfo["default_samplerate"]) if devinfo["default_samplerate"] else source.samplerate
        native_channels = int(devinfo["max_input_channels"])
        if native_channels < 1:
            raise RuntimeError(f"Selected device has no input channels: {source.label}")

        samplerates = []
        for sr in [native_sr, 48000, 44100, 16000]:
            if sr not in samplerates:
                samplerates.append(sr)

        channel_choices = []
        for ch in [native_channels, 1]:
            if ch >= 1 and ch not in channel_choices:
                channel_choices.append(ch)

        dtype_choices = ["float32", "int16"]

        is_wasapi = "WASAPI" in source.hostapi_name.upper()
        last_exc = None

        for dtype in dtype_choices:
            for sr in samplerates:
                for ch in channel_choices:
                    extra_settings = None
                    if is_wasapi:
                        try:
                            extra_settings = sd.WasapiSettings(
                                exclusive=False,
                                auto_convert=True,
                                explicit_sample_format=(dtype != "float32"),
                            )
                        except TypeError:
                            extra_settings = sd.WasapiSettings(
                                exclusive=False,
                                auto_convert=True,
                            )

                    try:
                        self.debug(
                            f"Trying sounddevice open: device={source.device_index}, "
                            f"sr={sr}, channels={ch}, dtype={dtype}, wasapi={is_wasapi}"
                        )

                        sd.check_input_settings(
                            device=source.device_index,
                            samplerate=sr,
                            channels=ch,
                            dtype=dtype,
                            extra_settings=extra_settings,
                        )

                        self.stream = sd.InputStream(
                            samplerate=sr,
                            channels=ch,
                            dtype=dtype,
                            callback=self._audio_callback,
                            blocksize=0,
                            device=source.device_index,
                            extra_settings=extra_settings,
                        )
                        self.stream.start()

                        self.capture_samplerate = sr
                        self.debug(
                            f"Opened sounddevice stream OK: device={source.device_index}, "
                            f"sr={sr}, channels={ch}, dtype={dtype}"
                        )
                        return

                    except Exception as e:
                        last_exc = e
                        self.debug(
                            f"sounddevice open failed: device={source.device_index}, "
                            f"sr={sr}, channels={ch}, dtype={dtype}, err={repr(e)}"
                        )
                        try:
                            if self.stream is not None:
                                self.stream.close()
                        except Exception:
                            pass
                        self.stream = None

        if is_wasapi:
            fb = self._find_fallback_non_wasapi_source(source)
            if fb is not None:
                self.debug(
                    f"WASAPI open failed for '{source.label}'. "
                    f"Trying fallback non-WASAPI source: '{fb.label}'"
                )
                return self._open_sounddevice_input(fb)

        raise last_exc if last_exc else RuntimeError("Unable to open input stream")

    def _pyaudio_callback(self, in_data, frame_count, time_info, status_flags):
        if not self._capture_accepting_audio():
            return (None, pyaudio.paComplete)

        try:
            audio = np.frombuffer(in_data, dtype=np.float32)

            selected_channels = self._selected_source_channels()
            if audio.size and selected_channels > 1 and audio.size % selected_channels == 0:
                audio = audio.reshape(-1, selected_channels)
                audio = self._downmix_to_mono(audio)

            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            with self.state_lock:
                if peak > self.audio_peak:
                    self.audio_peak = peak

            try:
                self.audio_queue.put_nowait(audio.copy())
            except queue.Full:
                pass

            now = time.time()
            if self.debug_enabled and (now - self.last_callback_debug_ts) >= 1.0:
                self.last_callback_debug_ts = now
                self.debug(
                    f"PyAudio loopback callback: frames={frame_count}, samples={audio.size}, peak={peak:.6f}"
                )

        except Exception as e:
            self.debug(f"_pyaudio_callback exception: {e}")

        return (None, pyaudio.paContinue)

    def _open_pyaudiowpatch_loopback(self, source: AudioSourceChoice):
        if not HAVE_PYAUDIOWPATCH:
            raise RuntimeError("PyAudioWPatch is not installed")

        if self.pa is None:
            self.pa = pyaudio.PyAudio()

        info = self.pa.get_device_info_by_index(source.device_index)
        self.debug(f"Selected PyAudio loopback info: {info}")

        channels = int(info.get("maxInputChannels", source.channels))
        if channels < 1:
            raise RuntimeError(f"Selected loopback device has no input channels: {source.label}")

        self.capture_samplerate = int(info.get("defaultSampleRate", source.samplerate))
        open_channels = channels

        self.debug(
            f"Opening PyAudioWPatch loopback: device={source.device_index}, "
            f"sr={self.capture_samplerate}, channels={open_channels}"
        )

        self.pa_stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=open_channels,
            rate=self.capture_samplerate,
            input=True,
            input_device_index=source.device_index,
            frames_per_buffer=1024,
            stream_callback=self._pyaudio_callback,
        )
        self.pa_stream.start_stream()

    def _on_rust_asio_chunk(self, audio: np.ndarray):
        if not self._capture_accepting_audio():
            return
        try:
            arr = np.asarray(audio, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                return
            peak = float(np.max(np.abs(arr))) if arr.size else 0.0
            with self.state_lock:
                if peak > self.audio_peak:
                    self.audio_peak = peak
            try:
                self.audio_queue.put_nowait(arr.copy())
            except queue.Full:
                pass
        except Exception as e:
            self.debug(f"_on_rust_asio_chunk exception: {e}")

    def _open_rust_asio_input(self, source: AudioSourceChoice):
        if not self.rust_asio.available():
            raise RuntimeError(f"Rust DLL unavailable: {self.rust_asio.load_error()}")

        selected_input_channels = self._get_selected_asio_channels()
        effective_channel_count = self._get_effective_asio_channel_count(source)
        self.capture_samplerate = int(source.samplerate) if source.samplerate else 48000

        requested_buffer_frames = self._get_selected_asio_buffer_frames(source)
        available_buffer_frames = self._enumerate_asio_buffer_frames(source)

        self.debug(
            f"Opening Rust ASIO: device_id='{source.device_id}', "
            f"sr={self.capture_samplerate}, requested_channels={effective_channel_count}, "
            f"selected_input_channels={selected_input_channels or 'Auto'}, "
            f"requested_buffer_frames={requested_buffer_frames}, "
            f"driver_supported_frames={available_buffer_frames}"
        )

        start_result = {"done": False, "error": None}

        def _start_call():
            try:
                self.rust_asio.start(
                    device_id=source.device_id or source.label,
                    sample_rate=self.capture_samplerate,
                    channels=effective_channel_count,
                    buffer_frames=requested_buffer_frames,
                    input_channels=selected_input_channels if selected_input_channels else None,
                )
            except Exception as e:
                start_result["error"] = e
            finally:
                start_result["done"] = True

        start_thread = threading.Thread(target=_start_call, daemon=True, name="RustAsioStartCall")
        start_thread.start()

        start_deadline = time.time() + 20.0
        last_stage_logged = None
        while not start_result["done"]:
            if self.start_cancelled:
                self.ui_queue.put(("startup_status", "Startup: cancellation requested; waiting for native ASIO call to return"))
                raise RuntimeError("Rust ASIO startup cancelled")
            stage = self._safe_rust_asio_last_error() or "startup pending"
            self.asio_runtime_var.set(f"ASIO Runtime: startup pending | {stage}")
            self.ui_queue.put(("startup_status", f"Startup: pending | stage={stage} | Stop cancels the wait, but native ASIO may take a moment to return"))
            if stage != last_stage_logged:
                self.debug(f"Rust ASIO startup stage: {stage}")
                last_stage_logged = stage
            if time.time() >= start_deadline:
                raise RuntimeError(f"Rust ASIO startup timeout. Last stage: {stage}")
            time.sleep(0.2)

        if start_result["error"] is not None:
            raise start_result["error"]

        runtime = self.rust_asio.runtime_info()
        if runtime.sample_rate > 0:
            self.capture_samplerate = runtime.sample_rate
        actual_frames = self.rust_asio.buffer_size_query()
        if actual_frames <= 0:
            actual_frames = requested_buffer_frames
        actual_channels = runtime.channels or effective_channel_count
        actual_ms = self._format_buffer_ms_from_frames(actual_frames, self.capture_samplerate) if actual_frames > 0 else "n/a"

        try:
            driver_info = self.rust_asio.driver_info()
        except Exception as e:
            driver_info = {"driver_info_error": str(e)}

        if actual_frames > 0:
            if actual_frames not in self.asio_buffer_options:
                self.asio_buffer_options = sorted(set(self.asio_buffer_options + [actual_frames]))
                sr = max(1, int(self.capture_samplerate or 48000))
                self.asio_buffer_combo["values"] = ["Auto"] + [f"{frames} frames ({self._format_buffer_ms_from_frames(frames, sr)})" for frames in self.asio_buffer_options]
            self.asio_buffer_choice_var.set(f"{actual_frames} frames ({actual_ms})")
        self.asio_runtime_var.set(
            f"ASIO Runtime: {self.capture_samplerate} Hz | {actual_channels} ch | "
            f"{actual_frames} frames ({actual_ms})"
        )
        self.ui_queue.put(("startup_status", "Startup: native stream opened; starting reader and post-start diagnostics"))
        self.debug(f"ASIO driver info: {driver_info}")
        self.debug(f"ASIO runtime info: {runtime}")
        self.debug(f"ASIO runtime snapshot: {self._safe_rust_asio_runtime_snapshot()}")

        self.rust_reader = ThreadedAsioReader(
            self.rust_asio,
            mono=False,
            channels=actual_channels,
            max_read_samples=8192,
            on_chunk=self._on_rust_asio_chunk,
        )
        self.rust_reader.start()

        def _post_start_diag_worker():
            try:
                diag = self.rust_asio.refresh_diagnostics()
                self.debug(f"ASIO post-start diagnostics: {diag}")
            except Exception as e:
                self.debug(f"ASIO post-start diagnostics failed: {e}")

        threading.Thread(target=_post_start_diag_worker, daemon=True, name="RustAsioPostStartDiag").start()

    # --------------------------------------------------------
    # AUDIO CAPTURE
    # --------------------------------------------------------

    def start_capture(self):
        if self.running:
            self.debug("Start ignored: already running")
            return
        if self.start_in_progress:
            self.debug("Start ignored: startup already in progress")
            return

        if not self.model_loaded:
            messagebox.showwarning("Model", "Load the faster-whisper model first.")
            self.debug("Start failed: model not loaded")
            return

        source = self._get_selected_source()
        if not source:
            messagebox.showwarning("Source", "Select an input source first.")
            self.debug("Start failed: no source selected")
            return

        self.start_in_progress = True
        self.start_cancelled = False
        self.current_audio_chunks.clear()
        self.current_chunk_started_at = time.time()
        self.last_asr_metrics = None
        with self.state_lock:
            self.audio_peak = 0.0

        try:
            while True:
                self.audio_queue.get_nowait()
        except queue.Empty:
            pass

        self.debug(f"Attempting start_capture with source={source}")
        self.status_var.set("Starting audio backend...")
        self._set_startup_status("Startup: pending | initializing backend")
        self._refresh_control_states()

        def _startup_worker():
            try:
                self.stream = None
                self.pa_stream = None
                self.rust_reader = None

                if self.start_cancelled:
                    raise RuntimeError("Startup cancelled")

                if source.backend == "sounddevice":
                    self._open_sounddevice_input(source)
                elif source.backend == "pyaudiowpatch":
                    self._open_pyaudiowpatch_loopback(source)
                elif source.backend == "rust_asio":
                    self._open_rust_asio_input(source)
                else:
                    raise RuntimeError(f"Unknown backend: {source.backend}")

                self.running = True

                if self.start_cancelled:
                    self.debug("Startup completed after cancellation request; cleaning up")
                    self._cleanup_audio_backends()
                    self.running = False
                    self.ui_queue.put(("status", "Stopped"))
                    return

                self.worker_thread = threading.Thread(target=self._asr_worker_loop, daemon=True)
                self.worker_thread.start()

                self.ui_queue.put(("status", f"Recording / transcribing... ({self.capture_samplerate} Hz, {source.backend})"))
                self.ui_queue.put(("startup_status", "Startup: complete | stream active"))
                self.debug("Audio stream started successfully")

            except Exception as e:
                self.running = False
                self._cleanup_audio_backends()
                self.asio_runtime_var.set("ASIO Runtime: n/a")
                self.last_asr_metrics = None
                self.debug(f"start_capture startup exception: {repr(e)}")
                self.ui_queue.put(("error", f"Audio Start Error:\n{e}"))
            finally:
                self.start_in_progress = False

        self.start_thread = threading.Thread(target=_startup_worker, daemon=True, name="AudioStartWorker")
        self.start_thread.start()

    def stop_capture(self):
        self.debug("Stopping capture")

        if self.start_in_progress and not self.running:
            self.start_cancelled = True
            self.status_var.set("Stopping (startup pending)...")
            self._set_startup_status("Startup: cancellation requested | waiting for native ASIO call to return")
            self.debug("Stop requested while startup pending")
            return

        if self.start_in_progress:
            self.start_cancelled = True
            self.debug("Stop requested while startup still in progress")

        self.running = False
        self._cleanup_audio_backends()

        self.asio_runtime_var.set("ASIO Runtime: n/a")
        self.last_asr_metrics = None
        self.status_var.set("Stopped")
        self._set_startup_status("Startup: idle")
        self._refresh_control_states()

    def _audio_callback(self, indata, frames, time_info, status):
        if not self._capture_accepting_audio():
            return

        if status:
            try:
                self.ui_queue.put_nowait(("status", f"Audio status: {status}"))
            except queue.Full:
                pass

        try:
            arr = np.asarray(indata)

            if arr.dtype == np.int16:
                arr = arr.astype(np.float32) / 32768.0
            else:
                arr = arr.astype(np.float32, copy=False)

            if arr.ndim == 2:
                if arr.shape[1] > 1:
                    mono = np.mean(arr, axis=1)
                else:
                    mono = arr[:, 0]
            else:
                mono = arr.reshape(-1)

            peak = float(np.max(np.abs(mono))) if mono.size else 0.0
            with self.state_lock:
                if peak > self.audio_peak:
                    self.audio_peak = peak

            try:
                self.audio_queue.put_nowait(mono.copy())
            except queue.Full:
                pass

            now = time.time()
            if self.debug_enabled and (now - self.last_callback_debug_ts) >= 1.0:
                self.last_callback_debug_ts = now
                self.debug(
                    f"Audio callback: frames={frames}, samples={mono.size}, peak={peak:.6f}"
                )

        except Exception as e:
            self.debug(f"_audio_callback exception: {e}")

    # --------------------------------------------------------
    # ASR PIPELINE
    # --------------------------------------------------------

    def _asr_worker_loop(self):
        self.debug("ASR worker loop started")

        while self.running:
            try:
                chunk = self.audio_queue.get(timeout=0.25)
                self.current_audio_chunks.append(chunk)
            except queue.Empty:
                pass

            if self.current_chunk_started_at is None:
                self.current_chunk_started_at = time.time()

            elapsed = time.time() - self.current_chunk_started_at
            chunk_seconds = float(self.chunk_seconds_var.get())
            if elapsed < chunk_seconds:
                continue

            if not self.current_audio_chunks:
                self.current_chunk_started_at = time.time()
                continue

            audio = np.concatenate(self.current_audio_chunks)
            self.current_audio_chunks.clear()
            self.current_chunk_started_at = time.time()

            duration = len(audio) / max(1, self.capture_samplerate)
            self.debug(
                f"Finalize chunk: samples={len(audio)}, duration={duration:.3f}s, "
                f"target_chunk={self.chunk_seconds_var.get():.1f}s, sr={self.capture_samplerate}"
            )

            if duration < MIN_CHUNK_SECONDS:
                self.debug(f"Chunk skipped: too short ({duration:.3f}s)")
                continue

            self._process_audio_chunk(audio)

        self.debug("ASR worker loop exited")

    def _downmix_to_mono(self, audio: np.ndarray) -> np.ndarray:
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 1:
            mono = arr
        elif arr.ndim == 2:
            ch = arr.shape[1]
            if ch <= 1:
                mono = arr[:, 0]
            elif ch == 2:
                mono = 0.5 * (arr[:, 0] + arr[:, 1])
            elif ch == 3:
                mono = 0.25 * arr[:, 0] + 0.25 * arr[:, 1] + 0.50 * arr[:, 2]
            elif ch == 4:
                mono = 0.30 * arr[:, 0] + 0.30 * arr[:, 1] + 0.20 * arr[:, 2] + 0.20 * arr[:, 3]
            elif ch == 5:
                mono = 0.20 * arr[:, 0] + 0.20 * arr[:, 1] + 0.30 * arr[:, 2] + 0.15 * arr[:, 3] + 0.15 * arr[:, 4]
            elif ch >= 6:
                weights = np.zeros(ch, dtype=np.float32)
                weights[0] = 0.18
                weights[1] = 0.18
                weights[2] = 0.28
                weights[3] = 0.04
                weights[4] = 0.16
                weights[5] = 0.16
                if ch > 6:
                    extra = ch - 6
                    weights[6:] = max(0.0, 0.04 / max(1, extra))
                mono = arr @ weights
            else:
                mono = np.mean(arr, axis=1)
        else:
            mono = arr.reshape(-1)

        mono = np.asarray(mono, dtype=np.float32)
        if mono.size:
            mono = mono - np.mean(mono, dtype=np.float64)
            peak = float(np.max(np.abs(mono)))
            if peak > 0.999:
                mono = mono / peak
        return mono.astype(np.float32, copy=False)

    def _resample_audio_hq(self, audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if src_sr == dst_sr or audio.size == 0:
            return audio.astype(np.float32, copy=False)

        if HAVE_SOXR:
            return soxr.resample(audio, src_sr, dst_sr, quality='HQ').astype(np.float32, copy=False)

        if HAVE_SCIPY_RESAMPLE:
            g = math.gcd(int(src_sr), int(dst_sr))
            up = int(dst_sr) // g
            down = int(src_sr) // g
            return resample_poly(audio, up, down).astype(np.float32, copy=False)

        duration = len(audio) / float(src_sr)
        if duration <= 0:
            return audio.astype(np.float32, copy=False)
        src_x = np.linspace(0.0, duration, num=len(audio), endpoint=False)
        dst_len = max(1, int(round(duration * dst_sr)))
        dst_x = np.linspace(0.0, duration, num=dst_len, endpoint=False)
        return np.interp(dst_x, src_x, audio).astype(np.float32)

    def _process_audio_chunk(self, audio: np.ndarray):
        try:
            assert self.model is not None

            orig_code = language_name_to_code(self.orig_lang_var.get())
            dest_code = language_name_to_code(self.dest_lang_var.get())

            src_sr = getattr(self, "capture_samplerate", ASR_SAMPLE_RATE)
            audio_16k = self._resample_audio_hq(audio, src_sr, ASR_SAMPLE_RATE)

            self.debug(
                f"Running ASR: src_sr={src_sr}, dst_sr={ASR_SAMPLE_RATE}, "
                f"in_samples={len(audio)}, out_samples={len(audio_16k)}, "
                f"orig_lang={orig_code}, dest_lang={dest_code}"
            )

            audio_seconds = len(audio_16k) / float(ASR_SAMPLE_RATE)

            t0 = time.perf_counter()
            segments, info = self.model.transcribe(
                audio_16k,
                beam_size=5,
                language=None if orig_code == "auto" else orig_code,
                vad_filter=VAD_FILTER,
                condition_on_previous_text=True,
            )

            seg_list = list(segments)
            asr_seconds = max(1e-9, time.perf_counter() - t0)
            rtf = asr_seconds / max(audio_seconds, 1e-9)
            x_realtime = audio_seconds / asr_seconds
            text = "".join(seg.text for seg in seg_list).strip()

            self.last_asr_metrics = {
                "audio_seconds": audio_seconds,
                "asr_seconds": asr_seconds,
                "rtf": rtf,
                "x_realtime": x_realtime,
            }

            self.debug(
                f"ASR result: detected_lang={getattr(info, 'language', 'unknown')}, "
                f"segments={len(seg_list)}, text_len={len(text)}, "
                f"audio_seconds={audio_seconds:.3f}, asr_seconds={asr_seconds:.3f}, "
                f"rtf={rtf:.3f}, x_realtime={x_realtime:.2f}"
            )

            if not text:
                self.debug("ASR produced empty text")
                return

            detected = getattr(info, "language", "unknown")
            self.ui_queue.put(("append_asr", f"[{detected}] {text}\n"))

            translated = self._translate_with_llama(text, orig_code, dest_code)
            if translated.strip():
                self.ui_queue.put(("append_mt", f"[{code_to_language_name(dest_code)}] {translated}\n"))
                self.debug(f"MT parsed text_len={len(translated)}")
            else:
                self.debug("MT returned empty parsed translation")

            if self.auto_reverse_var.get():
                self.ui_queue.put(("swap_langs", None))
                self.debug("Auto reverse direction applied")

        except Exception as e:
            self.debug(f"_process_audio_chunk exception: {repr(e)}")
            self.ui_queue.put(("error", f"ASR/MT error:\n{e}"))

    # --------------------------------------------------------
    # AUDIO METER
    # --------------------------------------------------------

    def _schedule_meter_update(self):
        self._update_meter()
        self.root.after(50, self._schedule_meter_update)

    def _update_meter(self):
        with self.state_lock:
            peak = self.audio_peak
            self.audio_peak *= 0.82

        peak = max(0.0, min(1.0, peak))
        meter_w = 420
        fill_w = int(meter_w * peak)
        self.meter_canvas.coords(self.meter_bar, 0, 0, fill_w, 20)
        self.meter_text_var.set(f"Peak: {peak:.4f}")

        gpu = self._get_gpu_stats()
        if gpu:
            self.gpu_name_var.set(f"GPU: {gpu['name']}")
            self.gpu_mem_var.set(
                f"GPU Mem: {gpu['used_gb']:.1f}/{gpu['total_gb']:.1f} GB "
                f"(free {gpu['free_gb']:.1f}, gpu {gpu['gpu_util']:.0f}%, mem {gpu['mem_util']:.0f}%)"
            )
        else:
            self.gpu_name_var.set("GPU: n/a")
            self.gpu_mem_var.set("GPU Mem: n/a")

        if self.capture_mode_var.get() == "ASIO (Rust FFI)" and self.rust_asio.available():
            if self.start_in_progress or not self.running or self.rust_reader is None:
                state_text = "starting" if self.start_in_progress else ("running" if self.running else "idle")
                if self.start_in_progress:
                    self.asio_runtime_var.set("ASIO Runtime: startup pending...")
                elif not self.running:
                    self.asio_runtime_var.set("ASIO Runtime: n/a")
                else:
                    self.asio_runtime_var.set(f"ASIO Runtime: active | {state_text}")
            else:
                try:
                    rt = self.rust_asio.runtime_info()
                    actual_frames = self.rust_asio.buffer_size_query()
                    sr = rt.sample_rate or self.capture_samplerate
                    actual_ms = self._format_buffer_ms_from_frames(actual_frames, sr) if actual_frames > 0 else "n/a"
                    avail = self.rust_asio.available_samples()
                    snapshot = self._safe_rust_asio_runtime_snapshot()
                    latency_in = snapshot.get("input_latency_frames", 0)
                    latency_out = snapshot.get("output_latency_frames", 0)
                    pos = snapshot.get("sample_position", 0)
                    driver_name = snapshot.get("driver_name") or snapshot.get("driver") or ""
                    self.asio_runtime_var.set(
                        f"ASIO Runtime: {sr} Hz | {rt.channels or self._get_effective_asio_channel_count()} ch | "
                        f"{actual_frames} frames ({actual_ms}) | avail {avail} | in/out lat {latency_in}/{latency_out} | pos {pos}"
                    )
                    if driver_name:
                        self.status_var.set(f"Recording / transcribing... ({driver_name})")
                except Exception:
                    pass

        if self.last_asr_metrics:
            m = self.last_asr_metrics
            self.gpu_speed_var.set(
                f"ASR Speed: {m['x_realtime']:.2f}x realtime | "
                f"RTF {m['rtf']:.3f} | {m['audio_seconds']:.2f}s audio in {m['asr_seconds']:.2f}s"
            )
        else:
            self.gpu_speed_var.set("ASR Speed: n/a")

    # --------------------------------------------------------
    # TEXT HELPERS
    # --------------------------------------------------------

    def _trim_text_widget(self, widget: tk.Text, max_lines: int = TEXT_MAX_LINES):
        try:
            line_count = int(widget.index("end-1c").split(".")[0])
            if line_count > max_lines:
                delete_to = line_count - max_lines
                widget.delete("1.0", f"{delete_to}.0")
        except Exception:
            pass

    def _append_and_scroll(self, widget: tk.Text, text: str):
        widget.insert(tk.END, text)
        self._trim_text_widget(widget)
        widget.see(tk.END)
        widget.update_idletasks()

    # --------------------------------------------------------
    # UI ACTIONS
    # --------------------------------------------------------

    def clear_asr_text(self):
        self.asr_text.delete("1.0", tk.END)
        self.asr_text.see(tk.END)

    def clear_mt_text(self):
        self.mt_text.delete("1.0", tk.END)
        self.mt_text.see(tk.END)

    def swap_languages(self):
        o = self.orig_lang_var.get()
        d = self.dest_lang_var.get()

        if d == "Auto":
            d = "English"

        self.orig_lang_var.set(d)
        self.dest_lang_var.set("English" if o == "Auto" else o)
        self.debug(f"Languages swapped: orig={self.orig_lang_var.get()}, dest={self.dest_lang_var.get()}")

    def _schedule_ui_pump(self):
        self._pump_ui_queue()
        self.root.after(100, self._schedule_ui_pump)

    def _pump_ui_queue(self):
        while True:
            try:
                kind, payload = self.ui_queue.get_nowait()
            except queue.Empty:
                break

            if kind == "append_asr":
                self._append_and_scroll(self.asr_text, payload)
            elif kind == "append_mt":
                self._append_and_scroll(self.mt_text, payload)
            elif kind == "append_debug":
                self._append_and_scroll(self.asr_text, payload)
            elif kind == "status":
                self.status_var.set(payload)
            elif kind == "startup_status":
                self._set_startup_status(payload)
            elif kind == "refresh_controls":
                self._refresh_control_states()
            elif kind == "swap_langs":
                self.swap_languages()
            elif kind == "error":
                self.status_var.set("Error")
                messagebox.showerror("Error", payload)

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    def close(self):
        self.stop_capture()
        try:
            if self.pa is not None:
                self.pa.terminate()
        except Exception:
            pass
        self.pa = None
        try:
            if self.nvml_initialized and HAVE_PYNVML:
                pynvml.nvmlShutdown()
        except Exception:
            pass
        self.nvml_initialized = False


# ============================================================
# ENTRY
# ============================================================

def main():
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass

    app = FasterWhisperTkApp(root)

    def on_close():
        app.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()