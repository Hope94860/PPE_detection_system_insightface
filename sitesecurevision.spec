# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Define data files to include
# Note: PyInstaller uses relative paths from the spec file location
added_files = [
    ('desktop_ui/assets', 'desktop_ui/assets'),
    ('models', 'models'),
    ('insightface_models', 'insightface_models'),
    ('yolov8n.onnx', '.'),
    ('yolov8n-pose.onnx', '.'),
    ('compass-connections.json', '.'),
]

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'api_server_n',
        'mongo_db_manager',
        'alert_engine',
        'buzzer_utils',
        'camera_detection_manager',
        'face_encoder',
        'face_recognizer',
        'incident_recorder',
        'inference_controller',
        'license_manager',
        'optimized_ppe_detection',
        'path_utils',
        'utils',
        'flask',
        'flask_cors',
        'pymongo',
        'onnxruntime',
        'cv2',
        'insightface',
        'numpy',
        'PIL',
        'PyQt6',
        'requests',
        'uuid',
        'cryptography',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SiteSecureVision',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, 
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SiteSecureVision',
)
