# -*- coding: utf-8 -*-
"""
KeenForge debug launcher: runs entry.py with stdout/stderr redirected to log
files while keeping a file-based __main__ module so multiprocessing.spawn
(child training processes) works correctly on Windows.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    # Redirect output BEFORE importing the app
    sys.stdout = open(os.path.join(_ROOT, "app_stdout.log"), "w", encoding="utf-8")
    sys.stderr = open(os.path.join(_ROOT, "app_stderr.log"), "w", encoding="utf-8")

    # Dump Python stack on native crash (segfault) to stderr log
    try:
        import faulthandler
        faulthandler.enable()
    except Exception:
        pass

    os.chdir(_ROOT)
    sys.argv = [os.path.join(_ROOT, "entry.py")]

    import runpy
    runpy.run_path(os.path.join(_ROOT, "entry.py"), run_name="__main__")
