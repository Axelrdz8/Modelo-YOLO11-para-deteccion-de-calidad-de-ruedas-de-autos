# Asegúrate de instalar la librería ultralytics si aún no la tienes
# Puedes instalarla con: pip install ultralytics

from ultralytics import YOLO
import cv2

# Cargar el modelo entrenado
model = YOLO("C:/Users/Axel/Desktop/Tareas Inteligencia Artificial/Modelo_yolo_ruedas_autos/runs/detect/train/weights/best.pt")

# Abrir un video o utilizar la webcam (cambiar 'ruta_al_video.mp4' si deseas usar un video)
cap = cv2.VideoCapture("C:/Users/Axel/Desktop/Tareas Inteligencia Artificial/Modelo_yolo_ruedas_autos/video_prueba.mp4")

# Obtener el ancho, alto y el frame rate del video original
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Definir el codec y crear un objeto VideoWriter para guardar el video
output_path = "C:/Users/Axel/Desktop/Tareas Inteligencia Artificial/Modelo_yolo_ruedas_autos/output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para el formato MP4
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar la detección con YOLO
    results = model.predict(frame, conf=0.80)

    # Dibujar las cajas de detección en la imagen
    annotated_frame = results[0].plot()

    # Guardar el frame anotado en el video de salida
    out.write(annotated_frame)

    # Mostrar el video con las detecciones
    cv2.imshow('YOLO Detections', annotated_frame)

    # Salir del bucle si presionas la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()