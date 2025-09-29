# crawl_backgrounds.py
# -*- coding: utf-8 -*-
from icrawler.builtin import GoogleImageCrawler
from pathlib import Path
import random, time, os, sys

# ====== CẤU HÌNH CHUNG ======
ROOT_DIR = Path("backgrounds")  # thư mục lưu ảnh
ROOT_DIR.mkdir(parents=True, exist_ok=True)
TOTAL_IMAGES = 1000  # tổng số ảnh muốn tải
MIN_SIZE = (512, 512)  # tối thiểu 512px để làm nền ghép
PER_KEYWORD_MIN = 40  # tối thiểu mỗi keyword
SLEEP_RANGE = (1, 1)  # delay giữa keyword
DOWNLOAD_THREADS = 3
PARSER_THREADS = 3

# Một số user-agent phổ biến để hạn chế bị chặn (xoay vòng ngẫu nhiên)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0 Safari/537.36",
]

# ====== BỘ TỪ KHÓA THEO NHÓM (VI + EN để phủ rộng) ======
KEYWORDS_GROUPS = {
    "outdoor": [
        "vỉa hè", "lề đường", "đường bê tông", "gạch lát vỉa hè",
        "bãi cỏ", "bãi đất trống", "bãi đậu xe", "công viên",
        "bến xe buýt", "kênh rạch bờ kè", "alley", "pavement",
        "parking lot", "sidewalk", "asphalt road", "public square"
    ],
    "indoor": [
        "hành lang trường học",
        "siêu thị lối đi", "warehouse floor", "workshop floor",
    ],
    "textures_surfaces": [
        "sàn gỗ", "gạch men", "bê tông xước", "đá granite",
        "inox surface", "metal texture", "plastic surface",
        "cardboard surface", "wooden table top"
    ],
    "camera_view": [
        "top-down floor", "top-down desk", "wide angle street",
        "telephoto street", "low angle pavement"
    ],
}

# Gộp & xáo trộn
ALL_KEYWORDS = []
for g in KEYWORDS_GROUPS.values():
    ALL_KEYWORDS.extend(g)
random.shuffle(ALL_KEYWORDS)

# Tính số ảnh mục tiêu cho mỗi keyword (chia đều theo nhóm)
K = len(ALL_KEYWORDS)
images_per_kw = max(PER_KEYWORD_MIN, TOTAL_IMAGES // K)
# điều chỉnh tổng gần bằng TOTAL_IMAGES
overshoot = images_per_kw * K - TOTAL_IMAGES

# Bỏ qua số ảnh dư ở một vài keyword cuối để gần sát TOTAL_IMAGES
skip_last = max(0, overshoot // images_per_kw) if overshoot > 0 else 0

# Đếm file sẵn có (để file_idx_offset không đè)
existing_files = len([p for p in ROOT_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
current_offset = existing_files

print(f"Will crawl ~{images_per_kw} images per keyword for {K} keywords "
      f"(target ≈ {TOTAL_IMAGES}). Existing files: {existing_files}")

# ====== CHẠY CRAWL ======
for idx, keyword in enumerate(ALL_KEYWORDS):
    if skip_last and idx >= K - skip_last:
        print(f"[SKIP] (balancing total) {keyword}")
        continue

    # nhỏ: giảm quota cho vài keyword ngẫu nhiên để tăng nhiễu phân bố
    kw_quota = images_per_kw - (1 if random.random() < 0.15 else 0)

    print(f"[{idx + 1}/{K}] Downloading {kw_quota:3d} images for: {keyword}")
    try:
        # Tạo crawler mới cho mỗi keyword với cấu hình riêng
        google_crawler = GoogleImageCrawler(
            storage={"root_dir": str(ROOT_DIR)},
            parser_threads=PARSER_THREADS,
            downloader_threads=DOWNLOAD_THREADS
        )

        # Set các thuộc tính downloader sau khi khởi tạo
        google_crawler.downloader.max_retry = 3
        google_crawler.downloader.timeout = 20
        google_crawler.downloader.user_agent = random.choice(USER_AGENTS)

        google_crawler.crawl(
            keyword=keyword,
            max_num=kw_quota,
            min_size=MIN_SIZE,
            file_idx_offset=current_offset,
            overwrite=False
        )
        current_offset += kw_quota

        if idx < K - 1:
            delay = random.randint(*SLEEP_RANGE)
            print(f"Sleeping {delay}s…")
            time.sleep(delay)

    except Exception as e:
        print(f"Error with keyword '{keyword}': {e}")
        continue

print("Crawl finished.")

# ====== LỌC SAU TẢI: loại GIF/ảnh quá mờ/ảnh trùng ======
print("Post-filtering (dedup & blur)…")
try:
    import numpy as np
    import cv2
    from PIL import Image
    import imagehash

    # 1) Xóa GIF & file rỗng
    for p in list(ROOT_DIR.glob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() in {".gif"} or p.stat().st_size < 10_000:
            p.unlink(missing_ok=True)


    # 2) Mờ (variance of Laplacian)
    def is_blurry(path, thr=90.0):
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return True
            fm = cv2.Laplacian(img, cv2.CV_64F).var()
            return fm < thr
        except:
            return True


    # 3) Dedup theo perceptual hash
    seen = set()
    removed_blur = removed_dup = 0
    for p in list(ROOT_DIR.glob("*")):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        try:
            if is_blurry(p):
                p.unlink(missing_ok=True)
                removed_blur += 1
                continue
            h = imagehash.phash(Image.open(str(p)).convert("RGB"))
            if h in seen:
                p.unlink(missing_ok=True)
                removed_dup += 1
            else:
                seen.add(h)
        except Exception:
            # lỗi đọc → xóa
            p.unlink(missing_ok=True)

    kept = len([1 for p in ROOT_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
    print(f"Removed blurry: {removed_blur}, dups: {removed_dup}. Kept: {kept} images.")

except ImportError as e:
    print(f"Post-filter skipped (install imagehash + opencv-python-headless to enable): {e}")
except Exception as e:
    print(f"Error during post-filtering: {e}")

print("Done.")