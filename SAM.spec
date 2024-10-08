# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import copy_metadata
datas = []
datas = copy_metadata('tqdm') + copy_metadata('regex') + copy_metadata('requests') + copy_metadata("packaging") + copy_metadata("filelock") + copy_metadata("numpy") + copy_metadata("tokenizers") + copy_metadata("huggingface-hub")+  copy_metadata("safetensors")+ copy_metadata('pyyaml')+copy_metadata('transformers')
datas += copy_metadata('importlib_metadata')
block_cipher = None


a = Analysis(
    ['D:\\SAM\\SAM.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['matplotlib.backends.backend_tkagg'],
    hookspath=['.'],
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
    name='train_fast_',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
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
    name='SAM',
)
