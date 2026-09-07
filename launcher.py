"""
SiteSecureVision — Application Launcher
========================================
Boot order
----------
1.  Check for ``sitesecurevision.lic``.
    • Missing  → start PyQt6 QApplication, show Setup Wizard.
    • Present  → validate MAC address.
      If MAC doesn't match → show error and exit.
2.  Start Flask backend in a daemon thread.
3.  Wait up to 300 s for the backend to become ready.
4.  Launch the main PyQt6 window.
"""

import sys
import os
import time
import threading
import multiprocessing
import argparse
from path_utils import get_app_data_dir

# ── Essential before any Qt import on Windows with PyInstaller ────────────────
multiprocessing.freeze_support()

BACKEND_URL = "http://127.0.0.1:5000/api/health"
# Global reference to main window to prevent garbage collection
_main_win = None

def find_mongo_bin():
    """Look for mongod.exe in common locations."""
    def_path = r"C:\Program Files\MongoDB\Server\8.2\bin\mongod.exe"
    if os.path.exists(def_path):
        return def_path
    
    # Check other versions
    base = r"C:\Program Files\MongoDB\Server"
    if os.path.exists(base):
        for v in os.listdir(base):
            p = os.path.join(base, v, "bin", "mongod.exe")
            if os.path.exists(p):
                return p
    return None

MONGO_BIN = find_mongo_bin()
def get_mongo_dbpath():
    """
    Determine the best path for MongoDB data.
    Order of preference:
    1. Local project 'data/db'
    2. Original developer data in 'C:\data\db' (Restoring older data)
    3. User AppData 'data/db' (Final deployed path)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_db = os.path.join(script_dir, "data", "db")
    
    # Check local project folder
    if os.path.exists(os.path.join(local_db, "WiredTiger")):
        print(f"📂 Using local project database: {local_db}")
        return local_db
    
    # Check for original developer data at C:\data\db
    older_db = r"C:\data\db"
    if os.path.exists(os.path.join(older_db, "WiredTiger")):
        print(f"📂 Restoring Original Developer Data from: {older_db}")
        return older_db
        
    # Default to User AppData directory (Standard for installations)
    return os.path.join(get_app_data_dir(), "data", "db")

MONGO_DBPATH = get_mongo_dbpath()

def _wait_for_mongo(timeout: int = 30) -> bool:
    """Poll until MongoDB is accepting connections or timeout expires."""
    from pymongo import MongoClient
    import pymongo.errors
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            client = MongoClient("mongodb://127.0.0.1:27017/", serverSelectionTimeoutMS=1000)
            client.admin.command('ping')
            client.close()
            return True
        except Exception:
            time.sleep(1)
    return False


def start_mongo():
    """Ensure MongoDB is running. Try Windows service first, then direct launch."""
    import subprocess

    # ── Step 1: Already running? ──────────────────────────────────────────────
    try:
        from pymongo import MongoClient
        client = MongoClient("mongodb://127.0.0.1:27017/", serverSelectionTimeoutMS=2000)
        client.admin.command('ping')
        client.close()
        print("✅ MongoDB is already running")
        return True
    except Exception:
        pass

    # ── Step 2: Try starting the Windows service (requires admin) ─────────────
    print("🚀 Attempting to start MongoDB Windows service...")
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Start-Service -Name MongoDB"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            print("⏳ Waiting for MongoDB service to be ready...")
            if _wait_for_mongo(timeout=20):
                print("✅ MongoDB service started successfully")
                return True
    except Exception as svc_err:
        print(f"⚠️  Could not start MongoDB service: {svc_err}")

    # ── Step 3: Fall back to launching mongod.exe directly ────────────────────
    print("🚀 Launching mongod.exe directly...")

    # Prefer app-data dbpath; fall back to C:\data\db (default mongo location)
    dbpath = MONGO_DBPATH
    fallback_dbpath = r"C:\data\db"
    try:
        os.makedirs(dbpath, exist_ok=True)
    except OSError:
        dbpath = fallback_dbpath
        os.makedirs(dbpath, exist_ok=True)

    if MONGO_BIN and os.path.exists(MONGO_BIN):
        logpath = os.path.join(get_app_data_dir(), "mongo.log")
        try:
            os.makedirs(os.path.dirname(logpath), exist_ok=True)
        except OSError:
            logpath = os.path.join(dbpath, "mongo.log")

        subprocess.Popen(
            [MONGO_BIN, "--dbpath", dbpath,
             "--logpath", logpath, "--logappend"],
            creationflags=subprocess.CREATE_NO_WINDOW,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"⏳ Waiting for mongod to be ready (dbpath: {dbpath})...")
        if _wait_for_mongo(timeout=30):
            print("✅ mongod started successfully")
            return True
        else:
            print("❌ mongod did not become ready in time")
            return False

    print("❌ mongod.exe not found — cannot start MongoDB")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Backend helpers
# ─────────────────────────────────────────────────────────────────────────────

def start_backend():
    while True:
        try:
            import api_server_n
            api_server_n.app.run(
                host="127.0.0.1",
                port=5000,
                debug=False,
                use_reloader=False
            )
        except Exception as e:
            print(f"❌ Backend crashed: {e}")
            import time
            time.sleep(3)
            print("🔄 Restarting backend...")


def wait_for_backend(timeout: int = 300) -> bool:
    import requests
    print("⏳ Waiting for backend to become ready…")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(BACKEND_URL, timeout=1)
            if r.status_code == 200:
                print("✅ Backend is ready")
                return True
        except Exception:
            pass
        time.sleep(1)
    print("❌ Backend did not start in time")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Main entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import traceback
    
    # ── Parse Arguments First ────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="SiteSecureVision Launcher")
    parser.add_argument("--init-db", action="store_true", help="Initialize the MongoDB database and collections and then exit")
    args, unknown = parser.parse_known_args()
    
    if args.init_db:
        print("🔧 Initializing Database Setup...")
        if not start_mongo():
            print("❌ Failed to start MongoDB for initialization.")
            sys.exit(1)
        try:
            from mongo_db_manager import FaceRecognitionDB
            # This will create database, collections, indexes, and default admin user if not exists
            db = FaceRecognitionDB()
            print("✅ Database initialized successfully with all collections and indexes.")
        except Exception as e:
            print(f"❌ Error during component initialization: {e}")
            sys.exit(1)
        sys.exit(0)
    
    def global_exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        print("❌ Uncaught exception:", exc_type, exc_value)
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        
        from PyQt6.QtWidgets import QApplication, QMessageBox
        if QApplication.instance():
            box = QMessageBox()
            box.setWindowTitle("Application Error")
            box.setIcon(QMessageBox.Icon.Warning)
            box.setText(f"An unexpected error occurred. The application will try to continue running.\n\nError: {exc_value}")
            box.setDetailedText("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
            box.exec()

    sys.excepthook = global_exception_handler

    from PyQt6.QtWidgets import QApplication, QMessageBox
    from PyQt6.QtGui import QIcon, QFontDatabase
    from PyQt6.QtCore import Qt

    from license_manager import is_configured, validate_mac
    from path_utils import get_resource_path

    # ── 1️⃣  Create Qt application ─────────────────────────────────────────
    app = QApplication(sys.argv)
    
    # ✅ HIGH-DPI SUPPORT (Must be set before any window or icon creation)
    app.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app.setWindowIcon(QIcon(get_resource_path("desktop_ui/assets/images/Logo.png")))

    # Load fonts
    for font_file in ("Inter-Regular.ttf", "Inter-SemiBold.ttf", "Inter-Bold.ttf"):
        QFontDatabase.addApplicationFont(
            get_resource_path(f"desktop_ui/assets/fonts/{font_file}")
        )

    # Load theme
    try:
        with open(get_resource_path("desktop_ui/assets/theme.qss"), "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
    except FileNotFoundError:
        print("[Theme] theme.qss not found — using defaults")

    # ── 2️⃣  First-run setup (no .lic file) ───────────────────────────────
    if not is_configured():
        print("🔧 No licence file found — showing Setup Wizard")

        # We need the backend up before the setup screen so cameras can be
        # registered immediately after config creation.
        _start_backend_thread()
        backend_ready = wait_for_backend()

        from desktop_ui.setup_config_screen import SetupConfigWindow

        setup_win = SetupConfigWindow()
        setup_win.showMaximized()

        # When setup is done, close the wizard and continue
        _done = [False]

        def _on_setup_complete():
            _done[0] = True
            setup_win.close()

        setup_win.setup_complete.connect(_on_setup_complete)

        # Block until setup window closes
        app.exec()

        if not _done[0]:
            # User closed the wizard — abort
            print("⚠  Setup cancelled by user — exiting")
            sys.exit(0)

        # Re-validate now that .lic has been written
        ok, msg = validate_mac()
        if not ok:
            _show_error(app, "Licence Error", msg)
            sys.exit(1)

        # Launch the main window in the *same* process (restart the event loop)
        _launch_main_window(app, backend_ready)
        sys.exit(app.exec())

    # ── 3️⃣  Normal start — validate MAC ──────────────────────────────────
    print("🔍 Validating licence…")
    ok, msg = validate_mac()
    if not ok:
        # Need the application to show an error dialog
        _show_error(app, "Licence Validation Failed", msg)
        sys.exit(1)

    print(f"✅ {msg}")

    # ── 3.5 Start MongoDB ───────────────────────────────────────────────────
    if not start_mongo():
        print("⚠️  Warning: Could not start MongoDB. The application may fail to connect to the database.")

    # ── 4️⃣  Start backend ─────────────────────────────────────────────────
    _start_backend_thread()
    if not wait_for_backend():
        _show_error(app, "Backend Error",
                    "The backend server could not be started within 300 seconds.\n"
                    "Please check your installation.")
        sys.exit(1)

    # ── 5️⃣  Launch main UI ────────────────────────────────────────────────
    _launch_main_window(app, True)
    sys.exit(app.exec())


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _start_backend_thread():
    t = threading.Thread(target=start_backend, daemon=True)
    t.start()
    return t


def _show_error(app, title: str, msg: str):
    from PyQt6.QtWidgets import QMessageBox
    box = QMessageBox()
    box.setWindowTitle(title)
    box.setIcon(QMessageBox.Icon.Critical)
    box.setText(msg)
    box.exec()


def _launch_main_window(app, backend_ready: bool):
    global _main_win
    from desktop_ui.main import MainWindow
    _main_win = MainWindow()
    _main_win.showMaximized()
    print("""
    ==========================================
    🚀 SiteSecureVision Started
    ==========================================
    """)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
