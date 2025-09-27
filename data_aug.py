import cv2
import numpy as np
import os
import random
import shutil
from typing import List, Tuple, Optional
from pathlib import Path


class YOLODataAugmentation:
    def __init__(self, background_dir: Optional[str] = None):
        """
        Khởi tạo class để tạo ảnh multi-object từ nhiều ảnh nguồn

        Args:
            background_dir: Đường dẫn đến thư mục chứa ảnh nền (tùy chọn)
        """
        self.background_dir = Path(background_dir) if background_dir else None
        self.background_images = self._load_background_images() if self.background_dir else []

    def _load_background_images(self) -> List[str]:
        """Tải danh sách đường dẫn ảnh nền"""
        if not self.background_dir or not self.background_dir.exists():
            return []

        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        backgrounds = []
        for ext in extensions:
            backgrounds.extend(self.background_dir.glob(f'*{ext}'))
            backgrounds.extend(self.background_dir.glob(f'*{ext.upper()}'))
        return [str(bg) for bg in backgrounds]

    def apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Áp dụng augmentation đơn giản cho object"""
        augmented = image.copy()

        # Horizontal flip
        if random.random() < 0.3:
            augmented = cv2.flip(augmented, 1)

        # Rotation
        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            h, w = augmented.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented = cv2.warpAffine(augmented, matrix, (w, h))

        # Scale
        if random.random() < 0.3:
            scale = random.uniform(0.8, 1.2)
            h, w = augmented.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            augmented = cv2.resize(augmented, (new_w, new_h))

        # Brightness and contrast
        if random.random() < 0.3:
            alpha = random.uniform(0.8, 1.2)  # contrast
            beta = random.randint(-20, 20)  # brightness
            augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)

        return augmented

    def read_yolo_label(self, label_path: str) -> List[Tuple[int, float, float, float, float]]:
        """Đọc file label YOLO"""
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        labels.append((class_id, x_center, y_center, width, height))
        return labels

    def yolo_to_bbox(self, yolo_coords: Tuple[float, float, float, float],
                     img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Chuyển đổi YOLO format sang pixel coordinates"""
        x_center, y_center, width, height = yolo_coords

        x_center_pixel = x_center * img_width
        y_center_pixel = y_center * img_height
        width_pixel = width * img_width
        height_pixel = height * img_height

        x1 = int(x_center_pixel - width_pixel / 2)
        y1 = int(y_center_pixel - height_pixel / 2)
        x2 = int(x_center_pixel + width_pixel / 2)
        y2 = int(y_center_pixel + height_pixel / 2)

        return max(0, x1), max(0, y1), x2, y2

    def bbox_to_yolo(self, bbox: Tuple[int, int, int, int],
                     img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """Chuyển đổi pixel coordinates sang YOLO format"""
        x1, y1, x2, y2 = bbox

        width_pixel = x2 - x1
        height_pixel = y2 - y1
        x_center_pixel = x1 + width_pixel / 2
        y_center_pixel = y1 + height_pixel / 2

        x_center = x_center_pixel / img_width
        y_center = y_center_pixel / img_height
        width = width_pixel / img_width
        height = height_pixel / img_height

        return x_center, y_center, width, height

    def extract_object(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Cắt object từ ảnh dựa trên bounding box"""
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        return image[y1:y2, x1:x2]

    def create_mask(self, object_img: np.ndarray) -> np.ndarray:
        """Tạo mask đơn giản cho object"""
        gray = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        return mask

    def check_overlap(self, new_bbox: Tuple[int, int, int, int],
                      existing_bboxes: List[Tuple[int, int, int, int]],
                      overlap_threshold: float = 0.3) -> bool:
        """Kiểm tra xem bbox mới có overlap với các bbox đã có không"""
        x1, y1, x2, y2 = new_bbox
        new_area = (x2 - x1) * (y2 - y1)

        for ex1, ey1, ex2, ey2 in existing_bboxes:
            # Tính intersection
            ix1 = max(x1, ex1)
            iy1 = max(y1, ey1)
            ix2 = min(x2, ex2)
            iy2 = min(y2, ey2)

            if ix1 < ix2 and iy1 < iy2:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                overlap_ratio = intersection / new_area
                if overlap_ratio > overlap_threshold:
                    return True
        return False

    def paste_object_on_background(self, object_img: np.ndarray, background_img: np.ndarray,
                                   existing_bboxes: List[Tuple[int, int, int, int]] = None,
                                   max_attempts: int = 50) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """
        Dán object lên nền mới với kiểm tra overlap

        Args:
            object_img: Ảnh object
            background_img: Ảnh nền
            existing_bboxes: Danh sách các bbox đã có
            max_attempts: Số lần thử tối đa để tìm vị trí không overlap

        Returns:
            Tuple (ảnh kết quả, bounding box mới hoặc None nếu không tìm được vị trí)
        """
        if existing_bboxes is None:
            existing_bboxes = []

        obj_h, obj_w = object_img.shape[:2]
        bg_h, bg_w = background_img.shape[:2]

        # Thử tìm vị trí không overlap
        for attempt in range(max_attempts):
            max_x = max(0, bg_w - obj_w)
            max_y = max(0, bg_h - obj_h)

            if max_x <= 0 or max_y <= 0:
                return background_img, None

            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            new_bbox = (x, y, x + obj_w, y + obj_h)

            # Kiểm tra overlap
            if not self.check_overlap(new_bbox, existing_bboxes):
                # Tạo mask cho object
                mask = self.create_mask(object_img)
                mask_3ch = cv2.merge([mask, mask, mask])

                # Tạo bản sao của nền
                result = background_img.copy()

                # Dán object lên nền
                roi = result[y:y + obj_h, x:x + obj_w]
                result[y:y + obj_h, x:x + obj_w] = np.where(mask_3ch > 0, object_img, roi)

                return result, new_bbox

        return background_img, None

    def collect_all_objects(self, input_images_dir: str, input_labels_dir: str) -> List[
        Tuple[str, int, Tuple[float, float, float, float]]]:
        """
        Thu thập tất cả objects từ dataset

        Returns:
            List of (image_path, class_id, yolo_coords)
        """
        images_dir = Path(input_images_dir)
        labels_dir = Path(input_labels_dir)

        all_objects = []

        # Tìm tất cả ảnh
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f'*{ext}'))
            image_files.extend(images_dir.glob(f'*{ext.upper()}'))

        for image_path in image_files:
            label_path = labels_dir / f"{image_path.stem}.txt"

            if label_path.exists():
                labels = self.read_yolo_label(str(label_path))
                for class_id, x_center, y_center, width, height in labels:
                    all_objects.append((str(image_path), class_id, (x_center, y_center, width, height)))

        return all_objects

    def create_multi_object_image(self, all_objects: List[Tuple[str, int, Tuple[float, float, float, float]]],
                                  output_path: str, target_size: Tuple[int, int] = (640, 640),
                                  max_objects: int = 8, min_objects: int = 2) -> bool:
        """
        Tạo một ảnh multi-object từ nhiều object khác nhau

        Args:
            all_objects: Danh sách tất cả objects có sẵn
            output_path: Đường dẫn output
            target_size: Kích thước ảnh đích
            max_objects: Số object tối đa trong một ảnh
            min_objects: Số object tối thiểu trong một ảnh

        Returns:
            True nếu tạo thành công
        """
        if len(all_objects) < min_objects:
            return False

        # Tạo hoặc chọn ảnh nền
        if self.background_images:
            background_path = random.choice(self.background_images)
            background = cv2.imread(background_path)
            if background is None:
                background = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 128
        else:
            # Tạo nền trơn màu ngẫu nhiên
            color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
            background = np.full((target_size[1], target_size[0], 3), color, dtype=np.uint8)

        background = cv2.resize(background, target_size)
        result_image = background.copy()

        # Chọn ngẫu nhiên số lượng objects
        num_objects = random.randint(min_objects, min(max_objects, len(all_objects)))
        selected_objects = random.sample(all_objects, num_objects)

        new_labels = []
        existing_bboxes = []

        for image_path, class_id, yolo_coords in selected_objects:
            try:
                # Đọc ảnh gốc
                source_image = cv2.imread(image_path)
                if source_image is None:
                    continue

                img_h, img_w = source_image.shape[:2]

                # Chuyển đổi sang pixel coordinates
                bbox = self.yolo_to_bbox(yolo_coords, img_w, img_h)

                # Cắt object
                object_img = self.extract_object(source_image, bbox)

                if object_img.size == 0:
                    continue

                # Apply augmentation cho object
                object_img = self.apply_augmentation(object_img)

                # Dán object lên nền
                result_image, new_bbox = self.paste_object_on_background(
                    object_img, result_image, existing_bboxes
                )

                if new_bbox is not None:
                    existing_bboxes.append(new_bbox)

                    # Chuyển đổi về YOLO format
                    yolo_coords_new = self.bbox_to_yolo(new_bbox, target_size[0], target_size[1])
                    new_labels.append((class_id, *yolo_coords_new))

            except Exception as e:
                print(f"Lỗi khi xử lý object từ {image_path}: {e}")
                continue

        if len(new_labels) < min_objects:
            return False

        # Lưu ảnh và label
        output_path = Path(output_path)
        output_image_path = output_path.with_suffix('.jpg')
        output_label_path = output_path.with_suffix('.txt')

        cv2.imwrite(str(output_image_path), result_image)

        # Lưu label
        with open(output_label_path, 'w') as f:
            for class_id, x_center, y_center, width, height in new_labels:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        return True

    def generate_multi_object_dataset(self, input_images_dir: str, input_labels_dir: str,
                                      output_dir: str, num_images: int = 100,
                                      target_size: Tuple[int, int] = (640, 640),
                                      max_objects: int = 8, min_objects: int = 2):
        """
        Tạo dataset multi-object với cấu trúc thư mục chuẩn YOLO

        Args:
            input_images_dir: Thư mục chứa ảnh gốc
            input_labels_dir: Thư mục chứa label
            output_dir: Thư mục output chính
            num_images: Số lượng ảnh multi-object cần tạo (có thể tạo bao nhiêu cũng được)
            target_size: Kích thước ảnh đích
            max_objects: Số object tối đa trong một ảnh
            min_objects: Số object tối thiểu trong một ảnh
        """
        print("Đang thu thập tất cả objects...")
        all_objects = self.collect_all_objects(input_images_dir, input_labels_dir)
        print(f"Tìm thấy {len(all_objects)} objects")

        if len(all_objects) < min_objects:
            print(f"Không đủ objects để tạo dataset (cần ít nhất {min_objects})")
            return

        # Tạo cấu trúc thư mục theo chuẩn YOLO
        output_dir = Path(output_dir)
        images_dir = output_dir / "images"
        labels_dir = output_dir / "labels"

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0
        attempt_count = 0
        max_attempts = num_images * 3  # Tối đa thử 3 lần số lượng ảnh cần tạo

        print(f"Bắt đầu tạo {num_images} ảnh multi-object...")

        while success_count < num_images and attempt_count < max_attempts:
            # Tạo tên file
            filename = f"multi_object_{success_count:04d}"
            image_path = images_dir / f"{filename}.jpg"
            label_path = labels_dir / f"{filename}.txt"

            if self.create_multi_object_image_v2(all_objects, str(image_path), str(label_path),
                                                 target_size, max_objects, min_objects):
                success_count += 1
                print(f"✓ Đã tạo: {filename} ({success_count}/{num_images})")
            else:
                print(f"✗ Thất bại lần thử {attempt_count + 1}")

            attempt_count += 1

        print(f"\n{'=' * 50}")
        print(f"Hoàn thành! Đã tạo {success_count}/{num_images} ảnh multi-object")
        print(f"Thư mục output:")
        print(f"  - Images: {images_dir}")
        print(f"  - Labels: {labels_dir}")
        print(f"{'=' * 50}")

    def create_multi_object_image_v2(self, all_objects: List[Tuple[str, int, Tuple[float, float, float, float]]],
                                     image_output_path: str, label_output_path: str,
                                     target_size: Tuple[int, int] = (640, 640),
                                     max_objects: int = 8, min_objects: int = 2) -> bool:
        """
        Tạo một ảnh multi-object với đường dẫn riêng biệt cho image và label

        Args:
            all_objects: Danh sách tất cả objects có sẵn
            image_output_path: Đường dẫn lưu ảnh
            label_output_path: Đường dẫn lưu label
            target_size: Kích thước ảnh đích
            max_objects: Số object tối đa trong một ảnh
            min_objects: Số object tối thiểu trong một ảnh

        Returns:
            True nếu tạo thành công
        """
        if len(all_objects) < min_objects:
            return False

        # Tạo hoặc chọn ảnh nền
        if self.background_images:
            background_path = random.choice(self.background_images)
            background = cv2.imread(background_path)
            if background is None:
                background = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 128
        else:
            # Tạo nền trơn màu ngẫu nhiên
            color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
            background = np.full((target_size[1], target_size[0], 3), color, dtype=np.uint8)

        background = cv2.resize(background, target_size)
        result_image = background.copy()

        # Chọn ngẫu nhiên số lượng objects
        num_objects = random.randint(min_objects, min(max_objects, len(all_objects)))
        selected_objects = random.sample(all_objects, num_objects)

        new_labels = []
        existing_bboxes = []

        for image_path, class_id, yolo_coords in selected_objects:
            try:
                # Đọc ảnh gốc
                source_image = cv2.imread(image_path)
                if source_image is None:
                    continue

                img_h, img_w = source_image.shape[:2]

                # Chuyển đổi sang pixel coordinates
                bbox = self.yolo_to_bbox(yolo_coords, img_w, img_h)

                # Cắt object
                object_img = self.extract_object(source_image, bbox)

                if object_img.size == 0:
                    continue

                # Apply augmentation cho object
                object_img = self.apply_augmentation(object_img)

                # Dán object lên nền
                result_image, new_bbox = self.paste_object_on_background(
                    object_img, result_image, existing_bboxes
                )

                if new_bbox is not None:
                    existing_bboxes.append(new_bbox)

                    # Chuyển đổi về YOLO format
                    yolo_coords_new = self.bbox_to_yolo(new_bbox, target_size[0], target_size[1])
                    new_labels.append((class_id, *yolo_coords_new))

            except Exception as e:
                print(f"Lỗi khi xử lý object từ {image_path}: {e}")
                continue

        if len(new_labels) < min_objects:
            return False

        # Lưu ảnh
        cv2.imwrite(image_output_path, result_image)

        # Lưu label
        with open(label_output_path, 'w') as f:
            for class_id, x_center, y_center, width, height in new_labels:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        return True

# Sử dụng
if __name__ == "__main__":
    # Thiết lập đường dẫn
    input_images_dir = "data/train/images"
    input_labels_dir = "data/train/labels"
    output_dir = "multi_object_data"
    background_dir = "backgrounds"  # Tùy chọn

    # Khởi tạo với hoặc không có background
    # augmenter = YOLODataAugmentation(background_dir)  # Hoặc None
    augmenter = YOLODataAugmentation()
    # Tạo dataset multi-object
    # augmenter.generate_multi_object_dataset(
    #     input_images_dir=input_images_dir,
    #     input_labels_dir=input_labels_dir,
    #     output_dir=output_dir,
    #     num_images=100,
    #     target_size=(640, 640),
    #     max_objects=6,
    #     min_objects=2
    # )

    augmenter.generate_multi_object_dataset(
        input_images_dir=input_images_dir,
        input_labels_dir=input_labels_dir,
        output_dir=output_dir,
        num_images=2000,  # Có thể tăng lên 1000, 2000... tùy ý
        target_size=(640, 640),
        max_objects=6,
        min_objects=2
    )