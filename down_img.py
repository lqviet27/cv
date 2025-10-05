#!/usr/bin/env python3

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List

from icrawler.builtin import GoogleImageCrawler

ROOT_DIR = Path("data_crawl")
ROOT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_SIZE = (512, 512)
BASELINE_TARGET_PER_CLASS = 1000
PER_KEYWORD_MIN = 40
SLEEP_RANGE = (1, 2)
DOWNLOAD_THREADS = 3
PARSER_THREADS = 3

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0 Safari/537.36",
]

CURRENT_CLASS_COUNTS: Dict[str, int] = {
    "cardboard_box": 516,
    "can": 784,
    "scrap_paper": 541,
    "plastic_bag": 548,
    "snack_bag": 667,
    "stick": 828,
    "plastic_cup": 550,
    "plastic_bottle_cap": 620,
    "plastic_box": 561,
    "battery": 337,
    "straw": 514,
    "chemical_spray_can": 372,
    "plastic_cup_lid": 580,
    "plastic_bottle": 390,
    "reuseable_paper": 419,
    "cardboard_bowl": 493,
    "light_bulb": 312,
}

CLASS_TARGET_OVERRIDE: Dict[str, int] = {}

CLASS_KEYWORDS: Dict[str, List[str]] = {
    "cardboard_box": [
        "several cardboard boxes on sidewalk",
        "a few cardboard boxes on warehouse floor",
        "multiple shipping cartons near door",
        "several cardboard boxes on street corner",
        "a couple of cardboard boxes on concrete floor",
    ],
    "can": [
        "several aluminum cans on street",
        "scattered soda cans on pavement",
        "a few crushed beverage cans on ground",
        "multiple beer cans near trash can",
        "three soda cans on sidewalk",
    ],
    "scrap_paper": [
        "scattered scrap paper on desk",
        "several paper scraps on floor",
        "a few waste paper sheets on table",
        "multiple scrap paper pieces on concrete",
        "several paper scraps near trash bin",
    ],
    "plastic_bag": [
        "several plastic bags on sidewalk",
        "a few plastic bags on grass",
        "multiple shopping bags on street",
        "several plastic bags on warehouse floor",
        "a couple of plastic bags near trash can",
    ],
    "snack_bag": [
        "several snack bags on sidewalk",
        "a few chip bags on ground",
        "multiple snack wrappers on street",
        "several crisp packets on pavement",
        "a couple of snack bags on bench",
    ],
    "stick": [
        "several wooden sticks on sidewalk",
        "a few twigs on pavement",
        "multiple sticks on dirt path",
        "several branches on concrete",
        "a couple of wooden sticks on grass",
    ],
    "plastic_cup": [
        "several plastic cups on sidewalk",
        "two or three plastic cups on table",
        "multiple plastic cups on ground",
        "a few plastic drink cups near trash can",
        "several disposable cups on floor",
    ],
    "plastic_bottle_cap": [
        "several plastic bottle caps on table",
        "a few bottle caps on ground",
        "multiple plastic caps on concrete",
        "scattered plastic lids on floor",
        "a couple of plastic bottle caps on pavement",
    ],
    "plastic_box": [
        "several plastic boxes on warehouse floor",
        "a few plastic containers on table",
        "multiple plastic storage bins on ground",
        "three plastic boxes on sidewalk",
        "several clear plastic boxes on concrete",
    ],
    "battery": [
        "several batteries on table",
        "a few AA batteries on floor",
        "multiple used batteries on pavement",
        "a couple of batteries on wooden desk",
        "several batteries on concrete",
    ],
    "straw": [
        "several plastic straws on table",
        "a few straws on sidewalk",
        "multiple drinking straws on ground",
        "scattered plastic straws on floor",
        "a couple of straws on concrete",
    ],
    "chemical_spray_can": [
        "several spray cans on floor",
        "multiple aerosol cans on sidewalk",
        "a few spray paint cans on ground",
        "several chemical spray cans in workshop",
        "a couple of aerosol cans on pavement",
    ],
    "plastic_cup_lid": [
        "several plastic cup lids on table",
        "a few coffee cup lids on ground",
        "multiple drink lids on sidewalk",
        "scattered plastic cup lids on floor",
        "a couple of plastic lids on concrete",
    ],
    "plastic_bottle": [
        "several plastic bottles on sidewalk",
        "a few plastic bottles on ground",
        "multiple water bottles near trash can",
        "two or three plastic bottles on pavement",
        "scattered plastic bottles on street",
    ],
    "reuseable_paper": [
        "several reusable paper bundles on desk",
        "a few stacks of recycled paper on table",
        "multiple reusable paper sheets on floor",
        "a couple of paper bundles on shelf",
        "several reusable paper stacks on office floor",
    ],
    "cardboard_bowl": [
        "several cardboard bowls on table",
        "a few paper bowls on ground",
        "multiple takeout bowls on counter",
        "a couple of compostable bowls on picnic table",
        "several cardboard bowls on desk",
    ],
    "light_bulb": [
        "several light bulbs on table",
        "a few bulbs on concrete floor",
        "multiple light bulbs on sidewalk",
        "a couple of broken light bulbs on ground",
        "several light bulbs on wooden desk",
    ],

}

ONLY_CLASSES: Iterable[str] | None = None  # e.g. ["plastic_cultery", "battery"] to target a subset


def count_images(directory: Path) -> int:
    return sum(1 for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def compute_target_for_class(class_name: str) -> int:
    desired_total = CLASS_TARGET_OVERRIDE.get(class_name, BASELINE_TARGET_PER_CLASS)
    current_samples = CURRENT_CLASS_COUNTS.get(class_name, 0)
    deficit = max(0, desired_total - current_samples)
    return deficit


def per_keyword_quota(remaining: int, keyword_count: int) -> int:
    if remaining <= 0:
        return 0
    keyword_count = max(keyword_count, 1)
    large_request = remaining >= PER_KEYWORD_MIN * keyword_count
    base = math.ceil(remaining / keyword_count)
    if large_request:
        base = max(base, PER_KEYWORD_MIN)
    return max(1, base)


def crawl_for_class(class_name: str, keywords: List[str], deficit: int) -> int:
    if deficit <= 0:
        print(f"[SKIP] {class_name}: already >= target (needs 0 more).")
        return 0

    class_dir = ROOT_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    existing_backgrounds = count_images(class_dir)
    remaining = max(0, deficit - existing_backgrounds)
    if remaining <= 0:
        print(f"[SKIP] {class_name}: {existing_backgrounds} backgrounds already cover deficit {deficit}.")
        return 0

    per_kw = per_keyword_quota(remaining, len(keywords))
    file_idx_offset = existing_backgrounds

    print(f"\n=== {class_name} ===")
    print(
        f"Dataset count: {CURRENT_CLASS_COUNTS.get(class_name, 0)}, target total: {CLASS_TARGET_OVERRIDE.get(class_name, BASELINE_TARGET_PER_CLASS)}, "
        f"needs {deficit} extra, existing backgrounds: {existing_backgrounds}, downloading ~{remaining} more (~{per_kw} per keyword)."
    )

    downloaded = 0
    for idx, keyword in enumerate(keywords, 1):
        if remaining <= 0:
            break

        quota = min(per_kw, remaining)
        user_agent = random.choice(USER_AGENTS)
        print(f"[{class_name}] ({idx}/{len(keywords)}) fetching {quota:3d} for '{keyword}'.")

        crawler = GoogleImageCrawler(
            storage={"root_dir": str(class_dir)},
            parser_threads=PARSER_THREADS,
            downloader_threads=DOWNLOAD_THREADS,
        )
        crawler.downloader.max_retry = 3
        crawler.downloader.timeout = 20
        crawler.downloader.user_agent = user_agent

        try:
            crawler.crawl(
                keyword=keyword,
                max_num=quota,
                min_size=MIN_SIZE,
                file_idx_offset=file_idx_offset,
                overwrite=False,
            )
        except Exception as exc:  # pragma: no cover - depends on network
            print(f"  ! Error while crawling '{keyword}': {exc}")
            continue

        file_idx_offset += quota
        remaining -= quota
        downloaded += quota

        if remaining > 0:
            delay = random.randint(*SLEEP_RANGE)
            if delay > 0:
                print(f"  Sleeping {delay}s before next keyword...")
                time.sleep(delay)

    print(f"[{class_name}] downloaded approximately {downloaded} files (remaining deficit {remaining}).")
    return downloaded


try:  # optional dependencies for post-filtering
    import numpy as np
    import cv2
    from PIL import Image
    import imagehash
except ImportError:  # pragma: no cover - runtime feature detection
    np = None
    cv2 = None
    Image = None
    imagehash = None


def is_blurry(path: Path, threshold: float = 90.0) -> bool:
    if np is None or cv2 is None:
        return False
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return True
        focus = cv2.Laplacian(img, cv2.CV_64F).var()
        return focus < threshold
    except Exception:
        return True


def clean_directory(directory: Path) -> tuple[int, int, int, int]:
    if Image is None or imagehash is None:
        return count_images(directory), 0, 0, 0

    removed_small = 0
    removed_blur = 0
    removed_dup = 0
    seen_hashes: set[str] = set()

    for path in list(directory.iterdir()):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        size_ok = path.stat().st_size >= 10_000

        if suffix == ".gif" or not size_ok:
            path.unlink(missing_ok=True)
            removed_small += 1
            continue

        if suffix not in IMAGE_EXTS:
            continue

        if is_blurry(path):
            path.unlink(missing_ok=True)
            removed_blur += 1
            continue

        try:
            with Image.open(path) as img:
                digest = str(imagehash.phash(img.convert("RGB")))
        except Exception:
            path.unlink(missing_ok=True)
            removed_small += 1
            continue

        if digest in seen_hashes:
            path.unlink(missing_ok=True)
            removed_dup += 1
        else:
            seen_hashes.add(digest)

    kept = count_images(directory)
    return kept, removed_blur, removed_dup, removed_small


def post_filter(selected_classes: Iterable[str]) -> None:
    if np is None or cv2 is None or Image is None or imagehash is None:
        print("Post-filter skipped (install numpy, opencv-python-headless, Pillow, imagehash).")
        return

    total_kept = 0
    total_blur = 0
    total_dup = 0
    total_small = 0

    for class_name in selected_classes:
        class_dir = ROOT_DIR / class_name
        if not class_dir.exists():
            continue
        kept, blur, dup, small = clean_directory(class_dir)
        total_kept += kept
        total_blur += blur
        total_dup += dup
        total_small += small
        print(f"[POST] {class_name}: kept {kept}, removed blur {blur}, dup {dup}, small {small}.")

    print(
        "Post-filter summary: kept {kept} images, removed {blur} blurry, {dup} duplicates, {small} tiny/bad files.".format(
            kept=total_kept, blur=total_blur, dup=total_dup, small=total_small
        )
    )


def main() -> None:
    random.seed()
    target_classes = list(CLASS_KEYWORDS.keys()) if not ONLY_CLASSES else list(ONLY_CLASSES)

    downloaded_total = 0
    processed_classes: List[str] = []
    for class_name in target_classes:
        keywords = CLASS_KEYWORDS.get(class_name, [])
        if not keywords:
            print(f"[WARN] No keywords configured for {class_name}, skipping.")
            continue
        deficit = compute_target_for_class(class_name)
        downloaded_total += crawl_for_class(class_name, keywords, deficit)
        if deficit > 0:
            processed_classes.append(class_name)

    print(f"\nRequested classes downloaded roughly {downloaded_total} new files in total.")
    post_filter(processed_classes)
    print("Done.")


if __name__ == "__main__":
    main()
