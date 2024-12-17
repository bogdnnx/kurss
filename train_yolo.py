from ultralytics import YOLO
import cv2
import os

# Пути к данным и модели
DATASET_PATH = "/home/boga/PycharmProjects/kurs5sem/Parking_space.v22i.yolov8/data.yaml"  # Замените на путь к вашему data.yaml
TEST_IMAGES_PATH = "/home/boga/PycharmProjects/kurs5sem/Parking_space.v22i.yolov8/test/images"  # Путь к тестовым изображениям
OUTPUT_PATH = "/home/boga/PycharmProjects/kurs5sem/Parking_space.v22i.yolov8/res"  # Путь для сохранения изображений с предсказаниями


# Шаг 1: Обучение модели
def train_yolo_model(data_path, model_name="yolov8s.pt", epochs=100, img_size=640, experiment_name="parking_detection"):
    model = YOLO(model_name)
    model.train(data=data_path, epochs=epochs, imgsz=img_size, name=experiment_name)
    print("Обучение завершено!")
    return model


# Шаг 2: Тестирование модели на изображениях
def test_yolo_model(model, source_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    results = model.predict(source=source_path, save=True, save_txt=True, project=output_path)
    print("Тестирование завершено. Результаты сохранены в:", output_path)
    return results


# Шаг 3: Визуализация результатов
def visualize_results(image_path, predictions_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for image_name in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path, image_name))
        prediction_path = os.path.join(predictions_dir, image_name.replace(".jpg", ".txt"))

        if os.path.exists(prediction_path):
            with open(prediction_path, "r") as f:
                for line in f:
                    cls, x_center, y_center, w, h = map(float, line.strip().split())
                    x1 = int((x_center - w / 2) * img.shape[1])
                    y1 = int((y_center - h / 2) * img.shape[0])
                    x2 = int((x_center + w / 2) * img.shape[1])
                    y2 = int((y_center + h / 2) * img.shape[0])
                    color = (0, 255, 0)  # Зеленый цвет
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, f"Class {int(cls)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Сохраняем визуализированное изображение
        output_file = os.path.join(output_dir, image_name)
        cv2.imwrite(output_file, img)

    print(f"Результаты визуализации сохранены в: {output_dir}")


# Шаг 4: Основной скрипт
if __name__ == "__main__":
    # 1. Обучение модели
    model = train_yolo_model(DATASET_PATH, epochs=50, img_size=640)

    # 2. Тестирование модели
    test_yolo_model(model, TEST_IMAGES_PATH, OUTPUT_PATH)

    # 3. Визуализация результатов
    visualize_results(TEST_IMAGES_PATH, os.path.join(OUTPUT_PATH, "predict", "labels"), "./visualized_results")
