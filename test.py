from ultralytics import YOLO
import os

# Пути к данным и модели
MODEL_PATH = "/home/boga/PycharmProjects/kurs5sem/runs/detect/parking_detection3/weights/best.pt"
NEW_TEST_IMAGES_PATH = "/home/boga/Загрузки/Car Parking Occupation.v1i.yolov8/test/images"
OUTPUT_PATH = "/home/boga/PycharmProjects/kurs5sem/new_results"


def test_yolo_model(model, source_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    results = model.predict(source=source_path, save=True, save_txt=True, project=output_path)

    for result in results:
        if result.boxes is None or not result.boxes.xyxy.numel():
            print(f"Нет предсказаний для изображения: {result.path}")
        else:
            print(f"Обработано изображение: {result.path}")

    print("Тестирование завершено. Результаты сохранены в:", output_path)


if __name__ == "__main__":
    # Загрузка модели
    model = YOLO(MODEL_PATH)

    # Тестирование модели
    test_yolo_model(model, NEW_TEST_IMAGES_PATH, OUTPUT_PATH)