# -*- coding: utf-8 -*-
"""Windows AppUserModelID helper.

Keeps the Windows taskbar identity correct: Explorer resolves the taskbar
button's display name and icon for an explicit AppUserModelID through
``HKCU\\Software\\Classes\\AppUserModelId\\<AUMID>`` (DisplayName + IconUri).
Without that registration the button falls back to the host process identity
(python.exe -> "Python" + a generic icon).

The entry scripts (entry.py / src/main.py) call ``SetCurrentProcessExplicitAppUserModelID``
with ``APP_ID`` and then ``register_app_id()`` before any UI is created.
"""

import os
import shutil
import sys

APP_ID = "com.keenforgeai.keenforge.v2.0"
APP_DISPLAY_NAME = "KeenForge"


def _project_root():
    # src/utils/app_id.py -> project root
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _stable_icon(src):
    """In frozen builds the bundled icon lives inside the PyInstaller temp
    dir; copy it to %LOCALAPPDATA% so the registry IconUri stays valid."""
    if not getattr(sys, "frozen", False):
        return src
    try:
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        dest_dir = os.path.join(base, "KeenForge")
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(src))
        if (not os.path.exists(dest)) or os.path.getmtime(src) > os.path.getmtime(dest):
            shutil.copyfile(src, dest)
        return dest
    except Exception:
        return src


def resolve_icon_path():
    """Return the best available app icon file (multi-size .ico preferred),
    or None when nothing is found. Path is absolute and CWD-independent."""
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", "")
        candidates = [
            os.path.join(base, "assets", "logo.ico"),
            os.path.join(base, "logo.png"),
        ]
    else:
        root = _project_root()
        candidates = [
            os.path.join(root, "assets", "logo.ico"),
            os.path.join(root, "logo.png"),
        ]
    for path in candidates:
        if os.path.exists(path):
            return _stable_icon(path)
    return None


def register_app_id():
    """Write DisplayName/IconUri for APP_ID under HKCU so the taskbar can
    resolve the app name and icon. Idempotent, safe to call on every start.
    Returns the resolved icon path (or None)."""
    icon = resolve_icon_path()
    try:
        import winreg

        key = winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER,
            r"Software\Classes\AppUserModelId\%s" % APP_ID,
            0,
            winreg.KEY_WRITE,
        )
        try:
            winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, APP_DISPLAY_NAME)
            if icon:
                winreg.SetValueEx(key, "IconUri", 0, winreg.REG_SZ, icon)
        finally:
            winreg.CloseKey(key)
    except Exception:
        pass
    return icon
