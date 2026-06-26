import os
import subprocess
import webbrowser
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
import time

try:
    import requests
except ImportError:
    requests = None


# =========================
# Config
# =========================
BASE_DIR = Path(r"C:\Plegma_Programming")
VENV_PY = BASE_DIR / "venv" / "Scripts" / "python.exe"

API_APP = "src.api_app:app"
API_HOST = "127.0.0.1"
API_PORT = 8001

UI_SCRIPT = BASE_DIR / "src" / "ui_app.py"
UI_HOST = "127.0.0.1"
UI_PORT = 7861  # must match ui_app.py launch port

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
API_LOG = LOG_DIR / "api_server.log"
UI_LOG = LOG_DIR / "ui_server.log"

api_proc = None
ui_proc = None


# =========================
# Helpers
# =========================
def _health_ok(timeout=0.8) -> bool:
    if requests is None:
        return False
    try:
        r = requests.get(f"http://{API_HOST}:{API_PORT}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _open_notepad(path: Path):
    os.system(f'notepad "{path}"')


def _start_process(cmd, cwd: Path, log_path: Path):
    log_f = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=str(cwd), stdout=log_f, stderr=log_f)


# =========================
# API controls
# =========================
def start_api():
    global api_proc

    if api_proc is not None and api_proc.poll() is None:
        messagebox.showinfo("PLEGMA Launcher", "API server is already running.")
        return

    if not VENV_PY.exists():
        messagebox.showerror("PLEGMA Launcher", f"Δεν βρέθηκε venv python:\n{VENV_PY}")
        return

    if _health_ok():
        messagebox.showinfo("PLEGMA Launcher", f"API already reachable at http://{API_HOST}:{API_PORT}")
        return

    cmd = [
        str(VENV_PY),
        "-m",
        "uvicorn",
        API_APP,
        "--host",
        API_HOST,
        "--port",
        str(API_PORT),
        "--reload",
    ]
    api_proc = _start_process(cmd, BASE_DIR, API_LOG)

    time.sleep(1.5)

    if api_proc.poll() is not None:
        messagebox.showerror("PLEGMA Launcher", f"API FAILED to start.\nΔες log:\n{API_LOG}")
        api_proc = None
        return

    if requests is not None:
        ok = _health_ok(timeout=1.2)
        if ok:
            messagebox.showinfo("PLEGMA Launcher", f"API started: http://{API_HOST}:{API_PORT}")
        else:
            messagebox.showwarning("PLEGMA Launcher", "API started αλλά /health δεν απαντά ακόμα.\nΠερίμενε λίγο.")


def stop_api():
    global api_proc

    if api_proc is not None and api_proc.poll() is None:
        api_proc.terminate()
        try:
            api_proc.wait(timeout=5)
        except Exception:
            api_proc.kill()
        api_proc = None
        messagebox.showinfo("PLEGMA Launcher", "API server stopped.")
        return

    messagebox.showinfo("PLEGMA Launcher", "API server is not running (from this launcher).")


# =========================
# UI controls (Gradio)
# =========================
def start_ui():
    global ui_proc

    if ui_proc is not None and ui_proc.poll() is None:
        messagebox.showinfo("PLEGMA Launcher", "UI is already running.")
        return

    if not VENV_PY.exists():
        messagebox.showerror("PLEGMA Launcher", f"Δεν βρέθηκε venv python:\n{VENV_PY}")
        return

    if not UI_SCRIPT.exists():
        messagebox.showerror("PLEGMA Launcher", f"Δεν βρέθηκε UI script:\n{UI_SCRIPT}")
        return

    cmd = [str(VENV_PY), str(UI_SCRIPT)]
    ui_proc = _start_process(cmd, BASE_DIR, UI_LOG)

    time.sleep(1.5)

    if ui_proc.poll() is not None:
        messagebox.showerror("PLEGMA Launcher", f"UI FAILED to start.\nΔες log:\n{UI_LOG}")
        ui_proc = None
        return

    messagebox.showinfo("PLEGMA Launcher", f"UI started: http://{UI_HOST}:{UI_PORT}")


def stop_ui():
    global ui_proc

    if ui_proc is not None and ui_proc.poll() is None:
        ui_proc.terminate()
        try:
            ui_proc.wait(timeout=5)
        except Exception:
            ui_proc.kill()
        ui_proc = None
        messagebox.showinfo("PLEGMA Launcher", "UI stopped.")
        return

    messagebox.showinfo("PLEGMA Launcher", "UI is not running (from this launcher).")


# =========================
# Open pages
# =========================
def open_swagger():
    webbrowser.open(f"http://{API_HOST}:{API_PORT}/docs")


def open_health():
    webbrowser.open(f"http://{API_HOST}:{API_PORT}/health")


def open_ui():
    webbrowser.open(f"http://{UI_HOST}:{UI_PORT}")


def open_api_log():
    _open_notepad(API_LOG)


def open_ui_log():
    _open_notepad(UI_LOG)


# =========================
# GUI
# =========================
root = tk.Tk()
root.title("PLEGMA Launcher")
root.geometry("460x260")
root.resizable(False, False)

tk.Label(root, text="PLEGMA Forecasting System", font=("Segoe UI", 13, "bold")).pack(pady=10)

frame = tk.Frame(root)
frame.pack(pady=5)

# Row 1
tk.Button(frame, text="Start API", width=18, command=start_api).grid(row=0, column=0, padx=8, pady=6)
tk.Button(frame, text="Stop API", width=18, command=stop_api).grid(row=0, column=1, padx=8, pady=6)
tk.Button(frame, text="Open Swagger", width=18, command=open_swagger).grid(row=0, column=2, padx=8, pady=6)

# Row 2
tk.Button(frame, text="Start UI", width=18, command=start_ui).grid(row=1, column=0, padx=8, pady=6)
tk.Button(frame, text="Stop UI", width=18, command=stop_ui).grid(row=1, column=1, padx=8, pady=6)
tk.Button(frame, text="Open UI", width=18, command=open_ui).grid(row=1, column=2, padx=8, pady=6)

# Row 3
tk.Button(frame, text="Open /health", width=18, command=open_health).grid(row=2, column=0, padx=8, pady=6)
tk.Button(frame, text="API Log", width=18, command=open_api_log).grid(row=2, column=1, padx=8, pady=6)
tk.Button(frame, text="UI Log", width=18, command=open_ui_log).grid(row=2, column=2, padx=8, pady=6)


def on_close():
    try:
        if ui_proc is not None and ui_proc.poll() is None:
            ui_proc.terminate()
        if api_proc is not None and api_proc.poll() is None:
            api_proc.terminate()
    except Exception:
        pass
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()