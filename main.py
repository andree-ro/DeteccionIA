import cv2
import mediapipe as mp
import numpy as np
from tkinter import filedialog
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
tiempo = time.time()
# Función para abrir un archivo desde el buscador de archivos
def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Cargar el archivo seleccionado como lienzo
        drawing_canvas = cv2.imread(file_path)
        return drawing_canvas
    else:
        return None

# Inicializar la captura de video desde la cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Esperar a que la cámara esté lista y leer el primer frame para obtener las dimensiones
ret, frame = cap.read()
if not ret:
    print("Error al acceder a la cámara")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Inicializar el canvas para dibujar con el mismo tamaño que el frame
height, width, _ = frame.shape
drawing_canvas = np.zeros((height, width, 3), dtype=np.uint8)

# Inicializar el módulo de manos
with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
    # Crear una ventana con tamaño autoajustable
    cv2.namedWindow('Hand Detection', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Voltear la imagen para que sea un espejo
        frame = cv2.flip(frame, 1)
        # Convertir la imagen a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Procesar la imagen y detectar las manos
        results = hands.process(frame_rgb)

        # Dibujar las marcas de las manos si se detectan
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Obtener la etiqueta de la mano (izquierda o derecha)
                hand_label = handedness.classification[0].label

                # Dibujar las marcas de la mano
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)
                )

                # Obtener las coordenadas del dedo índice (landmark 8) y del meñique (landmark 20)
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                x_index = int(index_finger_tip.x * frame.shape[1])
                y_index = int(index_finger_tip.y * frame.shape[0])
                x_pinky = int(pinky_tip.x * frame.shape[1])
                y_pinky = int(pinky_tip.y * frame.shape[0])

                # Si es la mano derecha y el dedo índice está hacia arriba y el meñique está abajo, dibujar
                if hand_label == 'Right' and index_finger_tip.y < pinky_tip.y and pinky_tip.y > 0.5:
                    # Cambiar el color del trazo a verde
                    color = (0, 255, 0)
                    # Dibujar el trazo
                    cv2.circle(drawing_canvas, (x_index, y_index), 5, color, -1)
                elif hand_label == 'Right' and index_finger_tip.y > pinky_tip.y and index_finger_tip.y > 0.5:
                    # Cambiar el color del trazo a rosado
                    color = (255, 0, 255)
                    # Dibujar el trazo
                    if pinky_tip.y < 0.5:  # Verificar si el meñique está levantado
                        cv2.circle(drawing_canvas, (x_pinky, y_pinky), 5, color, -1)

        # Redimensionar el frame y el lienzo a la mitad del ancho original
        frame_half = cv2.resize(frame, (width // 2, height))
        drawing_canvas_half = cv2.resize(drawing_canvas, (width // 2, height))

        # Concatenar el lienzo y el frame en una sola imagen
        combined_frame = np.concatenate((drawing_canvas_half, frame_half), axis=1)

        # Mostrar la imagen con las marcas de las manos y el dibujo
        cv2.imshow('Hand Detection', combined_frame)

        # Detectar si se presiona la tecla 'O' para abrir un archivo
        key = cv2.waitKey(1)
        if key == ord('o'):
            new_canvas = open_file()
            if new_canvas is not None:
                drawing_canvas = new_canvas

        # Salir del bucle al presionar 'ESC'
        if key == 27:  # 27 es el código de la tecla 'ESC'
            break

    # Definir la ruta donde se guardará el lienzo

    canvas_path = f'lienzo_dibujado{tiempo}.png'

    # Guardar el lienzo como una imagen
    cv2.imwrite(canvas_path, drawing_canvas)

    # Mostrar la ruta donde se guardó el lienzo
    print("Lienzo guardado en:", canvas_path)

    # Liberar el objeto de captura y cerrar todas las ventanas
    cap.release()
    cv2.destroyAllWindows()