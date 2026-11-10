"""Create a clean execution directory from the released final specification."""

from pathlib import Path
import argparse
import json
import shutil


PACKAGE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination",type=Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    destination.mkdir(parents=True,exist_ok=False)
    manifest = json.loads((PACKAGE / "freeze.json").read_text())
    paths = {row["path"] for row in manifest["files"]}
    paths.update(path.relative_to(PACKAGE).as_posix() for path in PACKAGE.glob("*.py"))
    paths.update(path.relative_to(PACKAGE).as_posix() for path in (PACKAGE / "verification").glob("*.py"))
    paths.update({"freeze.json","README.md","reproduce.sh","environment/python_requirements.txt"})
    paths.update({"derived_input_manifest.json","density_reference_definition378_v1.json","theory_density_reference378_v1.md"})
    paths.update({"supplement382_v1.json","theory/root_target_design382_v1.md","specifications/root_target_design382_v1.json"})
    derived = json.loads((PACKAGE / "derived_input_manifest.json").read_text())
    paths.update(row["path"] for row in derived["files"])
    for relative in sorted(paths):
        source = PACKAGE / relative
        target = destination / relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    print(destination)
    print("Prepared the frozen inputs and code in a new execution directory.")


if __name__ == "__main__":
    main()
