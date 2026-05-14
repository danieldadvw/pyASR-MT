# Requirements for:
# - pyVwxASRnMT_TESL_Canada_MIT_Licensed.py
# - rust_asio_portable.py
#
# Recommended Python: CPython 3.10 or 3.11 on Windows.
# tkinter is part of the standard Python installer on Windows and is not installed by pip.
# ctypes, json, os, queue, threading, time, dataclasses, typing, math, re are standard-library modules.

# Core numerical/runtime dependencies
numpy>=1.24
requests>=2.31

# ASR engine
faster-whisper>=1.0

# Audio input backends
sounddevice>=0.4.6
PyAudioWPatch>=0.2.12

# Optional high-quality resampling backends
# The code uses soxr first, then falls back to scipy.signal.resample_poly, then numpy interpolation.
soxr>=0.3.7
scipy>=1.10

# Optional NVIDIA GPU monitoring for VRAM-aware model status
nvidia-ml-py>=12.535
