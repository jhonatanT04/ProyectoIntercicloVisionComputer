import os
import subprocess
from flask import Flask, request, jsonify

app = Flask(__name__)

# Carpetas
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
BUILD_FOLDER = "build"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/procesar', methods=['POST'])
def procesar_archivo():
    if 'file' not in request.files:
        return jsonify({"error": "No se envió ningún archivo"}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    # Guardar archivo recibido
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    # Carpeta de salida específica para este archivo
    output_dir = os.path.join(OUTPUT_FOLDER, f"output_{os.path.splitext(file.filename)[0]}")
    os.makedirs(output_dir, exist_ok=True)

    # Ruta al ejecutable C++
    executable = os.path.join(BUILD_FOLDER, "main")

    if not os.path.exists(executable):
        return jsonify({"error": "El ejecutable main no existe en la carpeta build"}), 500

    # Ejecutar el programa C++
    try:
        result = subprocess.run(
            [executable, input_path, output_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        return jsonify({
            "message": "Procesamiento exitoso",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "output_folder": output_dir
        })

    except subprocess.CalledProcessError as e:
        return jsonify({
            "error": "Error al ejecutar el programa C++",
            "stdout": e.stdout,
            "stderr": e.stderr
        }), 500
from flask import send_from_directory

@app.route('/imagenes/<carpeta>', methods=['GET'])
def listar_imagenes(carpeta):
    # Carpeta real: outputs/output_L19 por ejemplo
    folder_path = os.path.join(OUTPUT_FOLDER, carpeta)

    if not os.path.exists(folder_path):
        return jsonify({"error": "La carpeta no existe"}), 404

    # Listar solo archivos de imagen
    imagenes = [
        f"http://localhost:5000/imagen/{carpeta}/{f}"
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]

    return jsonify({
        "carpeta": carpeta,
        "imagenes": imagenes
    })

@app.route('/imagen/<carpeta>/<filename>', methods=['GET'])
def obtener_imagen(carpeta, filename):
    folder_path = os.path.join(OUTPUT_FOLDER, carpeta)
    return send_from_directory(folder_path, filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
