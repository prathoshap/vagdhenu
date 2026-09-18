"""Backend selection for Vāgdhenu inference — CUDA, Apple Metal (MPS), or CPU.

Import this BEFORE torch. PyTorch reads PYTORCH_ENABLE_MPS_FALLBACK once, when the MPS
dispatch table is registered at `import torch`, so setting it afterwards has no effect.
The fallback matters because the mel front-end and the vocos ISTFT head both call FFT
kernels whose Metal coverage varies by torch build; without it a missing kernel raises
instead of quietly running on CPU.

Override with VAGDHENU_DEVICE=cpu|mps|cuda|cuda:1 to pin a backend by hand.
"""
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _prepend_venv_cuda_libs():
    """On Linux, prepend the venv's bundled NVIDIA CUDA libraries to LD_LIBRARY_PATH so torch's
    dlopen finds its own matched-version cuDNN/cuBLAS before the system's.

    Fixes a trap on AMIs that ship a system cuDNN at a different minor version than the one
    torch's wheel bundles (e.g. the AWS Deep Learning Base OSS AMI has cuDNN 9.10.2 on ldconfig,
    but torch 2.4.1+cu121 bundles cuDNN 9.24). Torch loads its own libcudnn.so.9 first via the
    wheel's rpath, then when it dlopens libcudnn_engines_runtime_compiled.so.9 the loader picks
    the system copy — version mismatch, CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED at first
    convolution. Prepending the venv path makes both dlopens hit the bundled copies.

    Only fires on Linux (macOS wheels don't ship these NVIDIA extras) and only for paths that
    actually exist. Must run before `import torch`; the loader re-reads LD_LIBRARY_PATH on each
    dlopen, but torch's own dlopens happen at import time.
    """
    if sys.platform != "linux":
        return
    site = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}",
                        "site-packages", "nvidia")
    if not os.path.isdir(site):
        return
    extras = [os.path.join(site, name, "lib") for name in ("cudnn", "cublas")]
    extras = [p for p in extras if os.path.isdir(p)]
    if not extras:
        return
    existing = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    # skip if the first extras entry is already at the front (idempotent across re-imports)
    if existing and existing[0] == extras[0]:
        return
    new = extras + [p for p in existing if p and p not in extras]
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(new)


_prepend_venv_cuda_libs()


def pick_device(prefer=None):
    """Return the torch device string to run on. `prefer` wins unless it is unavailable."""
    import torch

    requested = prefer or os.environ.get("VAGDHENU_DEVICE", "").strip()
    if requested and _available(requested):
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _available(name):
    import torch

    if name.startswith("cuda"):
        return torch.cuda.is_available()
    if name.startswith("mps"):
        return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    return name.startswith("cpu")


def describe(device):
    """One-line human summary of the chosen backend, for boot logs."""
    import torch

    if device.startswith("cuda"):
        return f"{device} ({torch.cuda.get_device_name(device if ':' in device else 0)})"
    if device.startswith("mps"):
        return "mps (Apple Metal — expect slower-than-realtime synthesis)"
    return "cpu (very slow; minutes per hemistich)"
