from pathlib import Path
import shutil
import kagglehub


DATASET = "sayedgamal99/smoke-fire-detection-yolo"


def main() -> None:
    print(f"Downloading dataset: {DATASET}")
    path = kagglehub.dataset_download(DATASET)

    source_path = Path(path)
    target_path = Path("data/raw/smoke-fire-detection-yolo")

    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        print(f"Dataset already copied to: {target_path.resolve()}")
        return

    shutil.copytree(source_path, target_path)
    print(f"Dataset copied to: {target_path.resolve()}")


if __name__ == "__main__":
    main()
