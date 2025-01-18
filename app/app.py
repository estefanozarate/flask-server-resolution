import os
import uuid
import threading
import torch
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify
import RRDBNet_arch as arch
import os.path as osp
from termcolor import colored

model_path = 'RRDB_ESRGAN_x4.pth'
device = torch.device('cpu')  # Cambiar a 'cuda' si tienes GPU y soporte CUDA en PyTorch

# Configuración de carpetas
ROOT_FOLDER = os.path.abspath("images")
ALLOWED_EXTENSIONS = {".jpg", ".png", ".jpeg"}
os.makedirs(ROOT_FOLDER, exist_ok=True)

images_HR = []
app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload_image_low_resolution():
    print(request.files)  # Debug: Verificar qué se recibe
    if "img" not in request.files:
        return jsonify({"ERROR": "Image not found"}), 400

    img = request.files["img"]
    if not img or img.filename == "":
        return jsonify({"ERROR": "Empty file"}), 400

    file_extension = os.path.splitext(img.filename)[1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        return jsonify({"ERROR": "Unsupported file type"}), 400

    session_id = str(uuid.uuid4())
    session_upload_folder = os.path.join(ROOT_FOLDER, session_id)
    os.makedirs(session_upload_folder, exist_ok=True)

    # Guardar archivo
    img_path = os.path.join(session_upload_folder, f"img_LR{file_extension}")
    img.save(img_path)

    print(colored(f"Image saved at {img_path}", "red"))

    thread_n = threading.Thread(target=super_resolution_img, args=(img_path, session_id))
    thread_n.start()

    return jsonify({"SUCCESS": f"Image saved and processing started at {img_path}"}), 200

def check_file(ruta_archivo):
    return os.path.isfile(ruta_archivo) and os.path.getsize(ruta_archivo) > 0

def super_resolution_img(image_path, session_id):
    print(colored(f"Processing image for super-resolution: {image_path}", "red"))
    
    # Cargar modelo
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)  # Define tu modelo RRDBNet correctamente
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    # Leer imagen
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = img * 1.0 / 255  # Normalizar a [0, 1]
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0).to(device)

    # Superresolución
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # RGB
    output = (output * 255.0).round().astype(np.uint8)  # Convierte a valores de píxeles [0, 255]

    output_path = f'images/{session_id}/img_HR.jpg'  # Cambia la extensión a .jpg
    cv2.imwrite(output_path, output, [cv2.IMWRITE_JPEG_QUALITY, 95])  # Calidad del JPEG ajustable
    print(f'DONE SUPERRESOLUTION: {output_path}')
    images_HR.append(str(output_path))


"""
# Cargar el modelo
model = arch.RRDBNet(3, 3, 64, 23, gc=32)  # Ajusta los parámetros si es necesario
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print(f'Model path {model_path}. \nProcessing image: {image_path}')

# Procesar la imagen
base = osp.splitext(osp.basename(image_path))[0]  # Obtén el nombre base del archivo sin extensión
img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Lee la imagen
if img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

img = img * 1.0 / 255  # Normaliza la imagen al rango [0, 1]
img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()  # Convierte a formato tensor
img_LR = img.unsqueeze(0).to(device)  # Agrega un batch dimension y mueve a dispositivo

# Realizar superresolución
with torch.no_grad():
    output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

# Convertir la salida de nuevo a formato de imagen
output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # RGB
output = (output * 255.0).round().astype(np.uint8)  # Convierte a valores de píxeles [0, 255]

# Guardar la imagen de salida en formato .jpg
output_path = f'/home/blackhat/Desktop/server_super_resolution/venv/src/{base}_rlt.jpg'  # Cambia la extensión a .jpg
cv2.imwrite(output_path, output, [cv2.IMWRITE_JPEG_QUALITY, 95])  # Calidad del JPEG ajustable
print(f'Result saved to {output_path}')

"""

@app.route("/images/<session_id>", methods=["GET"])
def get_super_resolution_images(session_id) -> None:
    
    img_path = os.path.join("images", session_id, "img_HR.jpg")
    if check_file(img_path):
        with open(img_path, "rb") as img_file:
            img_file_img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
            return jsonify({"frames": img_file_img_base64})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8880,debug=True)
