"""Write a portable equivalent of the WSL LIBERO config used for the preflight."""
import argparse
from importlib.util import find_spec
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Path for LIBERO_CONFIG_PATH")
    args = parser.parse_args()
    package = find_spec("libero.libero")
    if package is None or package.origin is None:
        parser.error("LIBERO must be installed in the current Python environment")
    root = Path(package.origin).resolve().parent
    settings = {
        "benchmark_root": root,
        "bddl_files": root / "bddl_files",
        "init_states": root / "init_files",
        "datasets": root.parent / "datasets",
        "assets": root / "assets",
    }
    args.directory.mkdir(parents=True, exist_ok=True)
    with (args.directory / "config.yaml").open("x", encoding="utf-8") as stream:
        for field, path in settings.items():
            stream.write(f'{field}: "{path.as_posix()}"\n')


if __name__ == "__main__":
    main()
