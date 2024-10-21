from ultralytics import YOLO

model = YOLO("yolo11n.pt")

if __name__ == '__main__':
    model.train(data="C:/Users/Axel/Desktop/Tareas Inteligencia Artificial/Modelo_yolo_ruedas_autos/data.yaml", epochs=100, imgsz=640)