from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import cv2
from ultralytics import YOLO

ImagePath = Union[str, Path]



def _display_image(image, title: str = "Prediction") -> None:
    try:
        from matplotlib import pyplot as plt

        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(title)
        plt.show()
    except Exception as exc:
        print(f"Matplotlib display failed ({exc}); falling back to OpenCV window.")
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, image)
        print("Press any key in the window to continue...")
        cv2.waitKey(0)
        cv2.destroyWindow(title)


def infer_single_image(
    model: YOLO,
    image_path: ImagePath,
    conf: float = 0.25,
    iou: float = 0.45,
    save_dir: Optional[Path] = None,
    display: bool = True,
) -> None:
    """Run inference on a single image and optionally save/display the result."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = model(str(image_path), conf=conf, iou=iou)
    annotated = results[0].plot()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{image_path.stem}_pred.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"Saved annotated image to: {out_path}")

    if display:
        _display_image(annotated, image_path.name)


def infer_folder(
    model: YOLO,
    folder_path: ImagePath,
    conf: float = 0.25,
    iou: float = 0.45,
    save_dir: Optional[Path] = None,
    display: bool = False,
    limit: Optional[int] = None,
) -> None:
    """Run inference on all images in a folder."""
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    image_paths: Iterable[Path] = [
        p for p in sorted(folder_path.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if limit is not None:
        image_paths = list(image_paths)[:limit]
    image_paths = list(image_paths)

    if not image_paths:
        print(f"No images found in {folder_path}")
        return

    if save_dir is None:
        save_dir = Path("runs") / "detect" / "inference"
    save_dir.mkdir(parents=True, exist_ok=True)

    from matplotlib import pyplot as plt  # local import

    for image_path in image_paths:
        results = model(str(image_path), conf=conf, iou=iou, verbose=False)
        annotated = results[0].plot()
        out_path = save_dir / f"{image_path.stem}_pred.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"Saved: {out_path}")

        if display:
            _display_image(annotated, image_path.name)


def infer_realtime(
    model: YOLO,
    source: Union[int, str] = 0,
    conf: float = 0.25,
    iou: float = 0.45,
    write_video: bool = False,
    output_path: Optional[Path] = None,
    window_name: str = "YOLOv8 Realtime",
) -> None:
    """Run realtime inference using a webcam index or video file."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")

    writer = None
    if write_video:
        if output_path is None:
            output_path = Path("runs") / "detect" / "realtime.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"Recording to: {output_path}")

    print("Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Stream ended or cannot fetch frame.")
                break

            results = model(frame, conf=conf, iou=iou, verbose=False)
            annotated = results[0].plot()

            if writer is not None:
                writer.write(annotated)

            cv2.imshow(window_name, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def _prompt_path(prompt: str, default: Optional[Path] = None, must_exist: bool = True) -> Path:
    while True:
        base = f"{prompt}"
        if default:
            base += f" [{default}]"
        base += ": "
        user_input = input(base).strip()
        if not user_input and default is not None:
            candidate = Path(default)
        else:
            candidate = Path(user_input)
        if must_exist and not candidate.exists():
            print(f"Path not found: {candidate}")
            continue
        return candidate


def _prompt_float(prompt: str, default: float) -> float:
    while True:
        val = input(f"{prompt} [{default}]: ").strip()
        if not val:
            return default
        try:
            return float(val)
        except ValueError:
            print("Please enter a valid number.")


def _prompt_int(prompt: str, default: int) -> int:
    while True:
        val = input(f"{prompt} [{default}]: ").strip()
        if not val:
            return default
        try:
            return int(val)
        except ValueError:
            print("Please enter a valid integer.")


def _prompt_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        val = input(f"{prompt} ({suffix}): ").strip().lower()
        if not val:
            return default
        if val in {"y", "yes"}:
            return True
        if val in {"n", "no"}:
            return False
        print("Please answer 'y' or 'n'.")


def _find_default_weights() -> Optional[Path]:
    runs_dir = Path("kaggle/working/runs")
    if runs_dir.exists():
        candidates = sorted(runs_dir.glob("detect/**/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]
    return None


def main() -> None:
    print("=== YOLOv8 Inference CLI ===")

    default_weights = _find_default_weights()
    weights_path = _prompt_path("Enter path to weights (.pt)", default=default_weights, must_exist=True)
    print(f"Loading model from {weights_path} ...")
    model = YOLO(str(weights_path))
    print("Model loaded.\n")

    while True:
        print("Choose an option:")
        print("  [1] Single image inference")
        print("  [2] Folder inference")
        print("  [3] Realtime inference (webcam / video)")
        print("  [q] Quit")
        choice = input("Your choice: ").strip().lower()

        if choice in {"q", "quit", "exit"}:
            print("Bye!")
            break
        if choice not in {"1", "2", "3"}:
            print("Invalid option. Please try again.\n")
            continue

        conf = _prompt_float("Confidence threshold", 0.25)
        iou = _prompt_float("IoU threshold", 0.45)

        if choice == "1":
            image_path = _prompt_path("Image path", must_exist=True)
            save = _prompt_yes_no("Save annotated image?", default=True)
            save_dir = None
            if save:
                save_dir = _prompt_path("Save directory", default=Path("runs") / "detect" / "inference", must_exist=False)
            display = _prompt_yes_no("Display result?", default=True)
            infer_single_image(model, image_path, conf=conf, iou=iou, save_dir=save_dir, display=display)

        elif choice == "2":
            folder_path = _prompt_path("Folder path", must_exist=True)
            save_dir = _prompt_path(
                "Directory to save outputs",
                default=Path("runs") / "detect" / "inference",
                must_exist=False,
            )
            display = _prompt_yes_no("Display images as they are processed?", default=False)
            limit_input = input("Limit number of images (leave blank for all): ").strip()
            limit = int(limit_input) if limit_input else None
            infer_folder(
                model,
                folder_path,
                conf=conf,
                iou=iou,
                save_dir=save_dir,
                display=display,
                limit=limit,
            )

        elif choice == "3":
            source_input = input("Video source (index or path) [0]: ").strip()
            if not source_input:
                source: Union[int, str] = 0
            else:
                try:
                    source = int(source_input)
                except ValueError:
                    source = source_input
            record = _prompt_yes_no("Record output to video?", default=False)
            output_path = None
            if record:
                output_path = _prompt_path(
                    "Video output path",
                    default=Path("runs") / "detect" / "realtime.mp4",
                    must_exist=False,
                )
            infer_realtime(
                model,
                source=source,
                conf=conf,
                iou=iou,
                write_video=record,
                output_path=output_path,
            )

        print()


if __name__ == "__main__":
    main()
