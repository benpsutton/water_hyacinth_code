from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
POINT_FILES_DIR = CONFIGS_DIR / "point_files"
INDICES_PATH = CONFIGS_DIR / "indices.json"
