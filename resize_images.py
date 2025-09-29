# resize_images.py
# -*- coding: utf-8 -*-
from PIL import Image, ImageOps
from pathlib import Path
import shutil
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# ====== CẤU HÌNH ======
INPUT_DIR = Path("backgrounds")  # Thư mục chứa ảnh gốc
OUTPUT_DIR = Path("backgrounds_640")  # Thư mục lưu ảnh đã resize
TARGET_SIZE = (640, 640)  # Kích thước đích
QUALITY = 95  # Chất lượng JPEG (1-100)
BACKUP_ORIGINAL = False  # True nếu muốn backup ảnh gốc
BACKUP_DIR = Path("backgrounds_original")  # Thư mục backup (nếu BACKUP_ORIGINAL=True)
NUM_WORKERS = mp.cpu_count()  # Số worker cho xử lý song song

# Các định dạng ảnh được hỗ trợ
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def resize_single_image(img_path, output_dir, target_size, quality, mode='smart'):
    """
    Resize một ảnh về kích thước target_size

    Modes:
    - 'smart': Cắt thông minh, giữ phần trung tâm quan trọng nhất
    - 'fit': Fit vừa khung, thêm padding nếu cần
    - 'stretch': Kéo giãn (không khuyến khích)
    """
    try:
        # Mở ảnh
        img = Image.open(img_path)

        # Chuyển về RGB nếu cần (để lưu JPEG)
        if img.mode in ('RGBA', 'LA', 'P'):
            # Tạo nền trắng cho ảnh có alpha channel
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if 'A' in img.mode else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize theo mode
        if mode == 'smart':
            # Smart crop: scale và cắt giữ phần trung tâm
            img = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)
        elif mode == 'fit':
            # Fit với padding
            img.thumbnail(target_size, Image.Resampling.LANCZOS)
            # Tạo canvas mới và paste ảnh vào giữa
            new_img = Image.new('RGB', target_size, (255, 255, 255))
            paste_x = (target_size[0] - img.width) // 2
            paste_y = (target_size[1] - img.height) // 2
            new_img.paste(img, (paste_x, paste_y))
            img = new_img
        else:  # stretch
            img = img.resize(target_size, Image.Resampling.LANCZOS)

        # Lưu ảnh
        output_path = output_dir / img_path.name
        # Đổi extension thành .jpg cho tất cả
        output_path = output_path.with_suffix('.jpg')
        img.save(output_path, 'JPEG', quality=quality, optimize=True)

        return True, img_path.name
    except Exception as e:
        return False, f"{img_path.name}: {str(e)}"


def process_batch(paths_batch, output_dir, target_size, quality):
    """Xử lý một batch ảnh"""
    results = []
    for path in paths_batch:
        results.append(resize_single_image(path, output_dir, target_size, quality))
    return results


def main():
    # Tạo thư mục output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Backup nếu cần
    if BACKUP_ORIGINAL:
        print(f"Backing up original images to {BACKUP_DIR}...")
        if BACKUP_DIR.exists():
            print(f"Backup directory {BACKUP_DIR} already exists. Skipping backup.")
        else:
            shutil.copytree(INPUT_DIR, BACKUP_DIR)
            print("Backup completed.")

    # Lấy danh sách ảnh cần xử lý
    image_paths = []
    for ext in SUPPORTED_FORMATS:
        image_paths.extend(INPUT_DIR.glob(f"*{ext}"))
        image_paths.extend(INPUT_DIR.glob(f"*{ext.upper()}"))

    if not image_paths:
        print(f"No images found in {INPUT_DIR}")
        return

    print(f"Found {len(image_paths)} images to process")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Target size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print(f"Using {NUM_WORKERS} workers")

    # Chia batch cho multi-processing
    batch_size = max(1, len(image_paths) // (NUM_WORKERS * 4))
    batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]

    # Process với progress bar
    success_count = 0
    error_count = 0
    errors = []

    with mp.Pool(NUM_WORKERS) as pool:
        process_func = partial(process_batch,
                               output_dir=OUTPUT_DIR,
                               target_size=TARGET_SIZE,
                               quality=QUALITY)

        with tqdm(total=len(image_paths), desc="Resizing images") as pbar:
            for batch_results in pool.imap_unordered(process_func, batches):
                for success, info in batch_results:
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                        errors.append(info)
                    pbar.update(1)

    # Báo cáo kết quả
    print(f"\n{'=' * 50}")
    print(f"✅ Successfully resized: {success_count} images")
    if error_count > 0:
        print(f"❌ Failed: {error_count} images")
        print("\nErrors:")
        for err in errors[:10]:  # Chỉ hiện 10 lỗi đầu
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

    # Thống kê dung lượng
    try:
        input_size = sum(f.stat().st_size for f in image_paths) / (1024 * 1024)
        output_files = list(OUTPUT_DIR.glob("*.jpg"))
        output_size = sum(f.stat().st_size for f in output_files) / (1024 * 1024)
        print(f"\n📊 Storage statistics:")
        print(f"  Input: {input_size:.1f} MB")
        print(f"  Output: {output_size:.1f} MB")
        print(f"  Saved: {input_size - output_size:.1f} MB ({(1 - output_size / input_size) * 100:.1f}%)")
    except:
        pass

    # Tùy chọn: Xóa thư mục gốc và đổi tên thư mục mới
    if success_count > 0 and error_count == 0:
        print(f"\n💡 Tip: All images processed successfully!")
        print(f"   You can now:")
        print(f"   1. Delete original folder: rm -rf {INPUT_DIR}")
        print(f"   2. Rename new folder: mv {OUTPUT_DIR} {INPUT_DIR}")

        # Uncomment để tự động thực hiện
        # response = input("\nDo you want to replace original folder? (y/n): ")
        # if response.lower() == 'y':
        #     shutil.rmtree(INPUT_DIR)
        #     OUTPUT_DIR.rename(INPUT_DIR)
        #     print("✅ Original folder replaced!")


if __name__ == "__main__":
    # Kiểm tra dependencies
    try:
        from tqdm import tqdm
    except ImportError:
        print("Please install tqdm: pip install tqdm")
        print("Running without progress bar...")


        # Simple fallback without tqdm
        class tqdm:
            def __init__(self, *args, **kwargs):
                self.total = kwargs.get('total', 0)
                self.desc = kwargs.get('desc', '')
                print(f"{self.desc}: Processing {self.total} items...")

            def update(self, n=1):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

    main()