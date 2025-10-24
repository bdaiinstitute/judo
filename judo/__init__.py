# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PACKAGE_ROOT / "models"
MESHES_PATH = MODEL_PATH / "meshes"
XML_PATH = MODEL_PATH / "xml"
DESCRIPTION_CACHE_DIR = "~/.cache/judo_descriptions_cache"
