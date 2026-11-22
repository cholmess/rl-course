"""Check that the course environment is installed correctly."""

import platform
import sys


def check(name, fn):
    try:
        detail = fn()
        print(f"{name:<22} {detail or ''} OK")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"{name:<22} FAILED: {exc}")
        return False


def python_version():
    v = sys.version_info
    if v < (3, 10):
        raise RuntimeError(f"Python {v.major}.{v.minor} found, need 3.10 or newer")
    return f"{v.major}.{v.minor}.{v.micro}"


def numpy_version():
    import numpy as np

    return np.__version__


def matplotlib_version():
    import matplotlib

    return matplotlib.__version__


def gymnasium_cartpole():
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=0)
    env.step(env.action_space.sample())
    env.close()
    return f"{gym.__version__}, CartPole obs shape {obs.shape}"


def torch_version():
    import torch

    x = torch.randn(3, requires_grad=True)
    (x**2).sum().backward()
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    return f"{torch.__version__}, device {device}"


if __name__ == "__main__":
    print(f"Platform: {platform.platform()}\n")
    results = [
        check("Python", python_version),
        check("NumPy", numpy_version),
        check("Matplotlib", matplotlib_version),
        check("Gymnasium", gymnasium_cartpole),
        check("PyTorch + autograd", torch_version),
    ]
    print("\nAll good. You're ready for module 0." if all(results) else "\nFix the FAILED lines, then run this again.")
