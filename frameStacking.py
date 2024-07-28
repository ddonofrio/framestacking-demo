import cv2
import numpy as np
from datetime import datetime
import os

# Constantes de configuración para los umbrales de color
COLOR_THRESHOLD_MIN = 40
COLOR_THRESHOLD_MAX = 255

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    if not ret:
        print("Error al cargar el video.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    accum_image = np.zeros_like(frame, dtype=np.float32)
    print("Creando imagen...")

    # Crear una carpeta para guardar las imágenes intermedias
    timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    output_folder = f'output_images_{timestamp}'
    os.makedirs(output_folder, exist_ok=True)

    progress_points = 0

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame.astype(np.float32)

        # Aplicar filtro: si el color está por debajo del umbral en cualquier canal, no se añade
        mask_min = np.all(frame >= COLOR_THRESHOLD_MIN, axis=2)
        
        # Verificar si cualquier canal en accum_image ha alcanzado el umbral máximo configurado
        mask_max = np.all(accum_image < COLOR_THRESHOLD_MAX, axis=2)
        
        # Solo se suman los píxeles que cumplen ambas condiciones
        combined_mask = np.logical_and(mask_min, mask_max)
        
        # Sumar el frame a la imagen acumulada solo en los píxeles válidos y que no sustraen
        for c in range(3):
            valid_pixels = combined_mask & (frame[:, :, c] >= accum_image[:, :, c])
            accum_image[:, :, c][valid_pixels] += frame[:, :, c][valid_pixels]
        
        # Asegurarse de que los valores no excedan COLOR_THRESHOLD_MAX
        np.clip(accum_image, 0, COLOR_THRESHOLD_MAX, out=accum_image)

        progress = (i + 1) / frame_count
        if progress >= progress_points / 100:
            if progress_points % 10 == 5:
                print('|', end='', flush=True)
            else:
                print('.', end='', flush=True)
            progress_points += 1

            accum_image_uint8 = accum_image.astype(np.uint8)
            intermediate_filename = os.path.join(output_folder, f'progress_image_{progress_points:03d}.png')
            cv2.imwrite(intermediate_filename, accum_image_uint8)

    accum_image = accum_image.astype(np.uint8)
    final_output_filename = os.path.join(output_folder, f'final_image_{timestamp}.png')
    cv2.imwrite(final_output_filename, accum_image)

    cap.release()
    cv2.destroyAllWindows()
    print("\nImagen creada.")
    
    # Crear el video timelapse
    create_timelapse(output_folder, 'timelapse_video.mp4', fps=15)

def create_timelapse(image_folder, output_video, fps):
    images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

process_video('ruta_al_video4.mp4')
