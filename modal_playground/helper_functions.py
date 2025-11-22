from pathlib import Path
import importlib

def check_requirements(req_file="requirements.txt"):
    installed = {}

    # Explicitly specify encoding
    content = Path(req_file).read_text(encoding="utf-16")  # <-- change from default

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        pkg_name = line.split("==")[0].split(">=")[0].split("<=")[0].split(">")[0].split("<")[0]

        name_map = {
            "Pillow": "PIL",
            "opencv-python": "cv2",
        }
        import_name = name_map.get(pkg_name, pkg_name)

        try:
            importlib.import_module(import_name)
            installed[pkg_name] = "OK"
        except ImportError as e:
            installed[pkg_name] = f"ERROR: {e}"

    return installed
