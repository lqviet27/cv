from ultralytics import YOLO
import cv2
import os

# Cấu hình đường dẫn
MODEL_PATH = r'E:\Workspace\Code\Python\CV_project\results\runs\detect\garbage_detect\weights\best.pt'


def improved_realtime_detection():
    """
    Version cải thiện để detect multiple objects
    """
    # Load model
    model = YOLO(MODEL_PATH)

    print("Classes trong model:", model.names)

    # Mở webcam
    cap = cv2.VideoCapture(0)

    print("Nhấn 'q' để thoát, 'i' để xem thông tin detection")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Predict với các tham số được điều chỉnh cho multiple detection
        results = model(
            frame,
            conf=0.3,  # Giảm confidence threshold
            iou=0.3,  # Giảm IoU threshold để tránh suppress các object gần nhau
            max_det=10,  # Tăng số lượng detection tối đa
            verbose=False
        )

        # Debug: In số lượng detections
        num_detections = len(results[0].boxes) if results[0].boxes is not None else 0

        # Vẽ kết quả
        annotated_frame = results[0].plot()

        # Thêm thông tin debug lên frame
        cv2.putText(annotated_frame, f"Detections: {num_detections}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Hiển thị
        cv2.imshow('Garbage Detection - Multiple Objects', annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('i'):
            # In thông tin chi tiết
            print_detection_info(results, model.names)

    cap.release()
    cv2.destroyAllWindows()


def print_detection_info(results, class_names):
    """
    In thông tin chi tiết về các detections
    """
    if results[0].boxes is not None:
        boxes = results[0].boxes
        print(f"\n=== Phát hiện {len(boxes)} objects ===")

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = class_names[cls]

            print(f"Object {i + 1}: {class_name}")
            print(f"  Confidence: {conf:.3f}")
            print(f"  Position: ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})")
            print(f"  Size: {x2 - x1:.0f} x {y2 - y1:.0f}")
    else:
        print("Không phát hiện object nào")


def advanced_realtime_detection():
    """
    Version nâng cao với nhiều tùy chọn
    """
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)

    # Các tham số có thể điều chỉnh
    conf_threshold = 0.25
    iou_threshold = 0.3
    max_detections = 15

    print("Controls:")
    print("'q' - Quit")
    print("'+'/'-' - Tăng/giảm confidence")
    print("'i'/'o' - Tăng/giảm IoU threshold")
    print("'d' - Debug info")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Predict với tham số động
        results = model(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            max_det=max_detections,
            verbose=False
        )

        # Vẽ manual để kiểm soát tốt hơn
        annotated_frame = draw_multiple_detections(frame.copy(), results, model.names)

        # Thêm thông tin tham số
        info_text = [
            f"Detections: {len(results[0].boxes) if results[0].boxes is not None else 0}",
            f"Conf: {conf_threshold:.2f}",
            f"IoU: {iou_threshold:.2f}"
        ]

        for i, text in enumerate(info_text):
            cv2.putText(annotated_frame, text, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Advanced Multiple Detection', annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            conf_threshold = min(0.9, conf_threshold + 0.05)
            print(f"Confidence: {conf_threshold:.2f}")
        elif key == ord('-'):
            conf_threshold = max(0.1, conf_threshold - 0.05)
            print(f"Confidence: {conf_threshold:.2f}")
        elif key == ord('i'):
            iou_threshold = min(0.9, iou_threshold + 0.05)
            print(f"IoU: {iou_threshold:.2f}")
        elif key == ord('o'):
            iou_threshold = max(0.1, iou_threshold - 0.05)
            print(f"IoU: {iou_threshold:.2f}")
        elif key == ord('d'):
            print_detection_info(results, model.names)

    cap.release()
    cv2.destroyAllWindows()


def draw_multiple_detections(frame, results, class_names):
    """
    Vẽ multiple detections với màu sắc khác nhau
    """
    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
    ]

    if results[0].boxes is not None:
        boxes = results[0].boxes

        for i, box in enumerate(boxes):
            # Lấy tọa độ
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # Lấy thông tin
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = class_names[cls]

            # Chọn màu (cycle through colors)
            color = colors[i % len(colors)]

            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Tạo label với ID
            label = f"#{i + 1} {class_name}: {conf:.2f}"

            # Vẽ background cho text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

            cv2.rectangle(frame, (x1, y1 - text_h - 10),
                          (x1 + text_w, y1), color, -1)

            # Vẽ text
            cv2.putText(frame, label, (x1, y1 - 5),
                        font, font_scale, (255, 255, 255), thickness)

    return frame


if __name__ == "__main__":
    improved_realtime_detection()