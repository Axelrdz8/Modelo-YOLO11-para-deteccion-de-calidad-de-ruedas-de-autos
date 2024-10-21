import os
import cv2
from ultralytics import YOLO

carpeta_imagenes = "C:/Users/Axel/Desktop/Tareas Inteligencia Artificial/Modelo_yolo_ruedas_autos/pruebas"
modelo = YOLO("C:/Users/Axel/Desktop/Tareas Inteligencia Artificial/Modelo_yolo_ruedas_autos/runs/detect/train/weights/best.pt")

imagenes = [f for f in os.listdir(carpeta_imagenes) if f.endswith('.jpeg')]

for imagen_nombre in imagenes:

    ruta_imagen = os.path.join(carpeta_imagenes, imagen_nombre)
    imagen = cv2.imread(ruta_imagen)
    
    #resultados = modelo(imagen)
    resultados = modelo.predict(imagen, conf = 0.6)
    
    imagen_con_resultados = resultados[0].plot()
    
    cv2.imshow('Detecciones_Ruedas', imagen_con_resultados)
    
    print(f'Mostrando {imagen_nombre}. Presiona cualquier tecla para continuar...')
    cv2.waitKey(0)

cv2.destroyAllWindows()