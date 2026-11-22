# Mac setup

About 20 minutes. Run these in Terminal.

## 1. Install the tools

If you don't have Homebrew yet, install it from [brew.sh](https://brew.sh). Then:

```bash
brew install git uv
```

`uv` is a fast Python installer and environment manager. It installs Python for you, so you don't need the system Python.

## 2. Clone the course

```bash
cd ~
git clone https://github.com/cholmess/rl-course.git
cd rl-course
```

If git asks for a password when you push, set up the GitHub CLI once with `brew install gh && gh auth login`.

## 3. Create the environment

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Run `source .venv/bin/activate` again whenever you open a new Terminal window for the course.

PyTorch uses your Mac's GPU through the `mps` backend if you have an Apple Silicon Mac (M1 or later). That's enough for phases 1 to 3.

## 4. Check it works

```bash
python modules/m00-setup/check_env.py
```

You should see every line end in `OK`.

## Editor

Any editor works. VS Code with the Python extension is a good default: open the `rl-course` folder and select the `.venv` interpreter when it asks.
