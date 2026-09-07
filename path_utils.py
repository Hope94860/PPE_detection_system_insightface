import os
import sys
import platform
from pathlib import Path

def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        # If not running as a bundled executable, use the current directory
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def get_app_data_dir() -> str:
    if platform.system() == "Windows":
        base_dir = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base_dir = Path.home() / ".config"
    app_dir = base_dir / "SiteSecureVision"
    app_dir.mkdir(parents=True, exist_ok=True)
    return str(app_dir)
