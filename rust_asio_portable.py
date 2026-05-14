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

import ctypes
import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np


@dataclass
class AsioDeviceInfo:
    id: str
    name: str
    sample_rate: int = 48000
    channels: int = 2
    driver: str = ""
    min_buffer_frames: int = 0
    preferred_buffer_frames: int = 0
    max_buffer_frames: int = 0
    buffer_granularity: int = 0
    available_input_channels: Optional[List[int]] = None
    dll_path: str = ""
    formats: Optional[List[str]] = None
    raw: Optional[dict] = None


@dataclass
class AsioRuntimeInfo:
    sample_rate: int = 0
    channels: int = 0
    buffer_frames: int = 0
    available_samples: int = 0
    ring_capacity_samples: int = 0
    peak_milli: int = 0
    dropped_samples: int = 0
    driver: str = ""
    driver_name: str = ""
    driver_dll_path: str = ""
    device_id: str = ""
    selected_input_channels: Optional[List[int]] = None
    input_latency_frames: int = 0
    output_latency_frames: int = 0
    sample_position: int = 0
    clock_source_index: int = -1
    status_text: str = ""
    stage: str = ""
    raw: Optional[dict] = None


class RustAsioBridge:
    """Portable ctypes wrapper for vwx_asio_bridge.dll (or the unified DLL).

    This FFI currently supports a single active ASIO stream per process.
    The Rust side uses global callback/stream state intentionally.
    """

    def __init__(self, dll_path: Optional[str] = None):
        self.dll_path = dll_path or self._default_dll_path()
        self.lib = None
        self._load_error = None
        self._bind_error = None
        self._try_load()

    def _default_dll_path(self) -> str:
        base = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base, "dw_asio_backend_ffi_unified1314.dll"),
            os.path.join(base, "target", "release", "dw_asio_backend_ffi_unified1314.dll"),
            os.path.join(base, "target", "debug", "dw_asio_backend_ffi_unified1314.dll"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return candidates[0]

    def _try_load(self):
        try:
            self.lib = ctypes.CDLL(self.dll_path)
            self._bind()
        except Exception as e:
            self.lib = None
            self._load_error = str(e)

    def _optional(self, name: str):
        try:
            return getattr(self.lib, name)
        except Exception:
            return None

    def _bind(self):
        try:
            self.vwx_get_last_error = self._optional("vwx_get_last_error")
            if self.vwx_get_last_error is not None:
                self.vwx_get_last_error.argtypes = []
                self.vwx_get_last_error.restype = ctypes.c_void_p

            self.vwx_free_string = self._optional("vwx_free_string")
            if self.vwx_free_string is not None:
                self.vwx_free_string.argtypes = [ctypes.c_void_p]
                self.vwx_free_string.restype = None

            self.vwx_asio_list_devices_json = self._optional("vwx_asio_list_devices_json")
            if self.vwx_asio_list_devices_json is not None:
                self.vwx_asio_list_devices_json.argtypes = []
                self.vwx_asio_list_devices_json.restype = ctypes.c_void_p

            self.vwx_asio_start = self._optional("vwx_asio_start")
            if self.vwx_asio_start is not None:
                self.vwx_asio_start.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
                self.vwx_asio_start.restype = ctypes.c_int

            self.vwx_asio_stop = self._optional("vwx_asio_stop")
            if self.vwx_asio_stop is not None:
                self.vwx_asio_stop.argtypes = []
                self.vwx_asio_stop.restype = ctypes.c_int

            self.vwx_asio_read_f32 = self._optional("vwx_asio_read_f32")
            if self.vwx_asio_read_f32 is not None:
                self.vwx_asio_read_f32.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_uint32]
                self.vwx_asio_read_f32.restype = ctypes.c_uint32

            self.vwx_asio_start_ex_json = self._optional("vwx_asio_start_ex_json")
            if self.vwx_asio_start_ex_json is not None:
                self.vwx_asio_start_ex_json.argtypes = [ctypes.c_char_p]
                self.vwx_asio_start_ex_json.restype = ctypes.c_int

            self.vwx_asio_get_runtime_json = self._optional("vwx_asio_get_runtime_json")
            if self.vwx_asio_get_runtime_json is not None:
                self.vwx_asio_get_runtime_json.argtypes = []
                self.vwx_asio_get_runtime_json.restype = ctypes.c_void_p

            self.vwx_asio_available_samples = self._optional("vwx_asio_available_samples")
            if self.vwx_asio_available_samples is not None:
                self.vwx_asio_available_samples.argtypes = []
                self.vwx_asio_available_samples.restype = ctypes.c_uint32

            self.vwx_asio_get_driver_info_json = self._optional("vwx_asio_get_driver_info_json")
            if self.vwx_asio_get_driver_info_json is not None:
                self.vwx_asio_get_driver_info_json.argtypes = []
                self.vwx_asio_get_driver_info_json.restype = ctypes.c_void_p

            self.vwx_asio_get_buffer_size = self._optional("vwx_asio_get_buffer_size")
            if self.vwx_asio_get_buffer_size is not None:
                self.vwx_asio_get_buffer_size.argtypes = []
                self.vwx_asio_get_buffer_size.restype = ctypes.c_uint32

            self.vwx_asio_refresh_diagnostics_json = self._optional("vwx_asio_refresh_diagnostics_json")
            if self.vwx_asio_refresh_diagnostics_json is not None:
                self.vwx_asio_refresh_diagnostics_json.argtypes = []
                self.vwx_asio_refresh_diagnostics_json.restype = ctypes.c_void_p
        except Exception as e:
            self._bind_error = str(e)

    def available(self) -> bool:
        return self.lib is not None and self._bind_error is None

    def load_error(self) -> str:
        return self._load_error or self._bind_error or "DLL not loaded"

    def _take_string(self, ptr) -> str:
        if not ptr:
            return ""
        try:
            raw = ctypes.cast(ptr, ctypes.c_char_p).value
            return raw.decode("utf-8", errors="replace") if raw else ""
        finally:
            if self.vwx_free_string is not None:
                self.vwx_free_string(ptr)

    def last_error(self) -> str:
        if not self.available():
            return self.load_error()
        if self.vwx_get_last_error is None:
            return "vwx_get_last_error export not found"
        return self._take_string(self.vwx_get_last_error())

    def _require_basic_asio(self):
        if not self.available():
            raise RuntimeError(self.load_error())
        missing = []
        if getattr(self, "vwx_asio_list_devices_json", None) is None:
            missing.append("vwx_asio_list_devices_json")
        if getattr(self, "vwx_asio_read_f32", None) is None:
            missing.append("vwx_asio_read_f32")
        if getattr(self, "vwx_asio_stop", None) is None:
            missing.append("vwx_asio_stop")
        if getattr(self, "vwx_asio_start", None) is None and getattr(self, "vwx_asio_start_ex_json", None) is None:
            missing.append("vwx_asio_start or vwx_asio_start_ex_json")
        if missing:
            raise RuntimeError("Missing ASIO exports: " + ", ".join(missing))

    def _json_from_ptr(self, ptr) -> dict | list:
        text = self._take_string(ptr)
        return json.loads(text) if text.strip() else {}

    def list_devices(self) -> List[AsioDeviceInfo]:
        self._require_basic_asio()
        raw = self._json_from_ptr(self.vwx_asio_list_devices_json())
        out: List[AsioDeviceInfo] = []
        for d in raw or []:
            out.append(AsioDeviceInfo(
                id=str(d.get("id") or d.get("name") or ""),
                name=str(d.get("name") or d.get("id") or "ASIO Device"),
                sample_rate=int(d.get("sample_rate", 48000) or 48000),
                channels=int(d.get("channels", 2) or 2),
                driver=str(d.get("driver") or d.get("name") or ""),
                min_buffer_frames=int(d.get("min_buffer_frames", 0) or 0),
                preferred_buffer_frames=int(d.get("preferred_buffer_frames", 0) or 0),
                max_buffer_frames=int(d.get("max_buffer_frames", 0) or 0),
                buffer_granularity=int(d.get("buffer_granularity", d.get("granularity", 0)) or 0),
                available_input_channels=list(d.get("available_input_channels") or []),
                dll_path=str(d.get("dll_path", "")),
                formats=[str(x) for x in (d.get("formats") or [])],
                raw=d,
            ))
        return out

    @staticmethod
    def enumerate_buffer_frame_choices(
        min_buffer_frames: int,
        preferred_buffer_frames: int,
        max_buffer_frames: int,
        buffer_granularity: int,
    ) -> List[int]:
        values = set()
        min_frames = max(0, int(min_buffer_frames or 0))
        preferred = max(0, int(preferred_buffer_frames or 0))
        max_frames = max(0, int(max_buffer_frames or 0))
        granularity = int(buffer_granularity or 0)

        if min_frames > 0:
            values.add(min_frames)
        if preferred > 0:
            values.add(preferred)
        if max_frames > 0:
            values.add(max_frames)

        if min_frames > 0 and max_frames >= min_frames:
            if granularity == 0:
                if preferred > 0:
                    values.add(preferred)
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

        return sorted(v for v in values if v > 0)

    def buffer_frame_choices(self, device: AsioDeviceInfo | dict | None) -> List[int]:
        if device is None:
            return []
        if isinstance(device, dict):
            min_frames = int(device.get("min_buffer_frames", 0) or 0)
            preferred_frames = int(device.get("preferred_buffer_frames", 0) or 0)
            max_frames = int(device.get("max_buffer_frames", 0) or 0)
            granularity = int(device.get("buffer_granularity", device.get("granularity", 0)) or 0)
        else:
            min_frames = int(device.min_buffer_frames or 0)
            preferred_frames = int(device.preferred_buffer_frames or 0)
            max_frames = int(device.max_buffer_frames or 0)
            granularity = int(device.buffer_granularity or 0)
        return self.enumerate_buffer_frame_choices(min_frames, preferred_frames, max_frames, granularity)

    def start(
        self,
        device_id: str,
        sample_rate: int,
        channels: int,
        buffer_frames: int = 1024,
        input_channels: Optional[Sequence[int]] = None,
    ):
        self._require_basic_asio()
        if self.vwx_asio_start_ex_json is not None:
            cfg = {
                "device_id": str(device_id),
                "sample_rate": int(sample_rate),
                "channels": int(channels),
                "buffer_frames": int(buffer_frames),
                "input_channels": [int(x) for x in (input_channels or [])],
            }
            rc = self.vwx_asio_start_ex_json(json.dumps(cfg).encode("utf-8"))
        elif self.vwx_asio_start is not None:
            rc = self.vwx_asio_start(
                str(device_id).encode("utf-8"),
                int(sample_rate),
                int(channels),
                int(buffer_frames),
            )
        else:
            raise RuntimeError("No compatible ASIO start export found")
        if rc != 0:
            raise RuntimeError(self.last_error() or f"ASIO start failed rc={rc}")

    def stop(self):
        self._require_basic_asio()
        rc = self.vwx_asio_stop()
        if rc != 0:
            raise RuntimeError(self.last_error() or f"ASIO stop failed rc={rc}")

    def read(self, max_samples: int = 8192) -> np.ndarray:
        self._require_basic_asio()
        buf = np.empty(int(max_samples), dtype=np.float32)
        n = int(self.vwx_asio_read_f32(buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), buf.size))
        return buf[:n].copy() if n > 0 else np.empty(0, dtype=np.float32)

    def available_samples(self) -> int:
        if self.vwx_asio_available_samples is not None:
            return int(self.vwx_asio_available_samples())
        return 0

    def runtime_snapshot(self) -> dict:
        if self.vwx_asio_get_runtime_json is not None:
            d = self._json_from_ptr(self.vwx_asio_get_runtime_json()) or {}
            return d if isinstance(d, dict) else {"raw": d}
        return {}

    def runtime_info(self) -> AsioRuntimeInfo:
        d = self.runtime_snapshot()
        if d:
            status_text = str(d.get("status_text", "") or "")
            return AsioRuntimeInfo(
                sample_rate=int(d.get("actual_sample_rate", d.get("sample_rate", 0)) or 0),
                channels=int(d.get("actual_channels", d.get("channels", 0)) or 0),
                buffer_frames=int(d.get("actual_buffer_frames", d.get("buffer_frames", 0)) or 0),
                available_samples=int(d.get("available_samples", 0) or 0),
                ring_capacity_samples=int(d.get("ring_capacity_samples", 0) or 0),
                peak_milli=int(d.get("peak_milli", 0) or 0),
                dropped_samples=int(d.get("dropped_samples", 0) or 0),
                driver=str(d.get("driver", d.get("driver_name", ""))),
                driver_name=str(d.get("driver_name", d.get("driver", ""))),
                driver_dll_path=str(d.get("driver_dll_path", "")),
                device_id=str(d.get("device_id", "")),
                selected_input_channels=[int(x) for x in (d.get("selected_input_channels") or [])],
                input_latency_frames=int(d.get("input_latency_frames", 0) or 0),
                output_latency_frames=int(d.get("output_latency_frames", 0) or 0),
                sample_position=int(d.get("sample_position", 0) or 0),
                clock_source_index=int(d.get("clock_source_index", -1) if d.get("clock_source_index") is not None else -1),
                status_text=status_text,
                stage=status_text,
                raw=d,
            )

        drv = self.driver_info()
        fallback_status = str(self.last_error() or "")
        return AsioRuntimeInfo(
            sample_rate=0,
            channels=0,
            buffer_frames=self.buffer_size_query(),
            available_samples=self.available_samples(),
            driver=str(drv.get("driver", drv.get("current_device_name", ""))),
            driver_name=str(drv.get("current_device_name", "")),
            driver_dll_path=str(drv.get("driver_dll_path", "")),
            input_latency_frames=int(drv.get("input_latency_frames", 0) or 0),
            output_latency_frames=int(drv.get("output_latency_frames", 0) or 0),
            status_text=fallback_status,
            stage=fallback_status,
            raw=drv if isinstance(drv, dict) else {"raw": drv},
        )

    def runtime_sample_rate(self) -> int:
        return self.runtime_info().sample_rate

    def runtime_channel_count(self) -> int:
        return self.runtime_info().channels

    def buffer_size_query(self) -> int:
        if self.vwx_asio_get_buffer_size is not None:
            return int(self.vwx_asio_get_buffer_size())
        return self.runtime_info().buffer_frames if self.vwx_asio_get_runtime_json is not None else 0

    def driver_info(self) -> dict:
        if self.vwx_asio_get_driver_info_json is not None:
            d = self._json_from_ptr(self.vwx_asio_get_driver_info_json())
            return d if isinstance(d, dict) else {"raw": d}
        info = self.runtime_info()
        return {
            "driver": info.driver,
            "current_device_name": info.driver_name,
            "driver_dll_path": info.driver_dll_path,
            "sample_rate": info.sample_rate,
            "channels": info.channels,
            "buffer_frames": info.buffer_frames,
            "input_latency_frames": info.input_latency_frames,
            "output_latency_frames": info.output_latency_frames,
        }

    def refresh_diagnostics(self) -> dict:
        if self.vwx_asio_refresh_diagnostics_json is not None:
            d = self._json_from_ptr(self.vwx_asio_refresh_diagnostics_json())
            return d if isinstance(d, dict) else {"raw": d}
        return self.driver_info()

    @staticmethod
    def downmix_to_mono(audio: np.ndarray, channels: Optional[int] = None, weights: Optional[Sequence[float]] = None) -> np.ndarray:
        arr = np.asarray(audio, dtype=np.float32)
        if arr.size == 0:
            return np.empty(0, dtype=np.float32)
        if arr.ndim == 1:
            if channels is None or channels <= 1:
                return arr.astype(np.float32, copy=False)
            if arr.size % channels != 0:
                raise ValueError(f"interleaved audio size {arr.size} is not divisible by channels={channels}")
            arr = arr.reshape(-1, channels)
        elif arr.ndim != 2:
            raise ValueError("audio must be 1D interleaved or 2D frames x channels")

        if arr.shape[1] == 1:
            return arr[:, 0].astype(np.float32, copy=False)

        if weights is None:
            mono = arr.mean(axis=1)
        else:
            w = np.asarray(weights, dtype=np.float32)
            if w.ndim != 1 or w.size != arr.shape[1]:
                raise ValueError("weights length must equal channel count")
            denom = float(w.sum())
            if abs(denom) < 1e-12:
                raise ValueError("weights sum cannot be zero")
            mono = (arr * w.reshape(1, -1)).sum(axis=1) / denom
        return mono.astype(np.float32, copy=False)


class ThreadedAsioReader:
    def __init__(
        self,
        bridge: RustAsioBridge,
        max_read_samples: int = 8192,
        mono: bool = False,
        channels: Optional[int] = None,
        weights: Optional[Sequence[float]] = None,
        poll_sleep_s: float = 0.002,
        max_queue_chunks: int = 256,
        on_chunk: Optional[Callable[[np.ndarray], None]] = None,
    ):
        self.bridge = bridge
        self.max_read_samples = int(max_read_samples)
        self.mono = bool(mono)
        self.channels = channels
        self.weights = list(weights) if weights is not None else None
        self.poll_sleep_s = float(poll_sleep_s)
        self.on_chunk = on_chunk
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue_chunks)
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_error: Optional[BaseException] = None

    @property
    def last_error(self) -> Optional[BaseException]:
        return self._last_error

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._last_error = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="RustAsioReader")
        self._thread.start()

    def stop(self, join_timeout: float = 1.0):
        self._stop_evt.set()
        t = self._thread
        if t is not None:
            t.join(timeout=join_timeout)

    def _run(self):
        while not self._stop_evt.is_set():
            try:
                avail = self.bridge.available_samples()
                if avail <= 0:
                    audio = self.bridge.read(self.max_read_samples)
                else:
                    audio = self.bridge.read(min(max(avail, 1), self.max_read_samples))
                if audio.size == 0:
                    time.sleep(self.poll_sleep_s)
                    continue

                if self.mono:
                    ch = self.channels or self.bridge.runtime_channel_count() or self.channels or 1
                    audio = self.bridge.downmix_to_mono(audio, channels=ch, weights=self.weights)

                if self.on_chunk is not None:
                    self.on_chunk(audio)
                else:
                    try:
                        self.queue.put_nowait(audio)
                    except queue.Full:
                        pass
            except BaseException as e:
                self._last_error = e
                time.sleep(min(max(self.poll_sleep_s, 0.01), 0.1))

    def read(self, timeout: Optional[float] = None) -> np.ndarray:
        return self.queue.get(timeout=timeout)

    def try_read(self) -> Optional[np.ndarray]:
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None
