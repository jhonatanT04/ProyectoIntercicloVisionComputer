import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os

# ==================== ARQUITECTURA DnCNN ====================
class DnCNN(nn.Module):
    """
    Red DnCNN para eliminación de ruido
    Arquitectura compatible con pesos pre-entrenados de KAIR
    Sin BatchNorm (versión original)
    """
    def __init__(self, channels=1, num_layers=20, num_features=64):
        super(DnCNN, self).__init__()
        
        layers = []
        # Primera capa: Conv + ReLU
        layers.append(nn.Conv2d(channels, num_features, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        
        # Capas intermedias: Conv + ReLU (sin BatchNorm)
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=True))
            layers.append(nn.ReLU(inplace=True))
        
        # Última capa: Solo Conv
        layers.append(nn.Conv2d(num_features, channels, kernel_size=3, padding=1, bias=True))
        
        self.dncnn = nn.Sequential(*layers)
    
    def forward(self, x):
        # El modelo predice el ruido, no la imagen limpia
        noise = self.dncnn(x)
        return x - noise  # Imagen limpia = imagen ruidosa - ruido estimado

# ==================== FUNCIONES DE UTILIDAD ====================
def cargar_imagen_ct(ruta_imagen):
    """
    Carga una imagen CT desde archivo
    Soporta PNG, JPEG, DICOM (.dcm, .IMA, etc)
    """
    # Intentar primero como DICOM (para .dcm, .IMA, etc)
    try:
        import pydicom
        ds = pydicom.dcmread(ruta_imagen)
        img = ds.pixel_array.astype(np.float32)
        print(f"✓ Imagen DICOM cargada: {img.shape}, dtype: {img.dtype}")
        return img
    except Exception as e:
        # Si falla, intentar como imagen estándar
        img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {ruta_imagen}\nError DICOM: {e}")
        img = img.astype(np.float32)
        print(f"✓ Imagen estándar cargada: {img.shape}")
        return img

def normalizar_imagen(img):
    """Normaliza la imagen al rango [0, 1]"""
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min > 0:
        img = (img - img_min) / (img_max - img_min)
    return img

def preprocesar_para_modelo(img):
    """
    Prepara la imagen para el modelo DnCNN
    Input: numpy array [H, W]
    Output: tensor [1, 1, H, W]
    """
    img_norm = normalizar_imagen(img)
    img_tensor = torch.from_numpy(img_norm).float()
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    return img_tensor

def posprocesar_resultado(tensor_salida, forma_original):
    """
    Convierte la salida del modelo a imagen numpy
    """
    img_salida = tensor_salida.squeeze().cpu().numpy()
    img_salida = np.clip(img_salida, 0, 1)  # Asegurar rango [0, 1]
    return img_salida

# ==================== FUNCIÓN PRINCIPAL ====================
def reducir_ruido_ct(ruta_imagen, mostrar_resultados=True):
    """
    Reduce el ruido de una imagen CT usando DnCNN pre-entrenado
    """
    print(f"Cargando imagen: {ruta_imagen}")
    
    # 1. Cargar imagen
    img_original = cargar_imagen_ct(ruta_imagen)
    print(f"Dimensiones de la imagen: {img_original.shape}")
    
    # 2. Crear y cargar modelo
    print("Inicializando modelo DnCNN...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
    modelo = DnCNN(channels=1, num_layers=20, num_features=64)
    modelo = modelo.to(device)
    
    # Cargar pesos pre-entrenados si existen
    peso_path = 'model_zoo/dncnn_gray_blind.pth'
    if os.path.exists(peso_path):
        print(f"✓ Cargando pesos pre-entrenados desde: {peso_path}")
        state_dict = torch.load(peso_path, map_location=device)
        
        # Ajustar nombres de las capas si es necesario
        # Los pesos usan 'model.X' pero nuestra arquitectura usa 'dncnn.X'
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_key = k.replace('model.', 'dncnn.')
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        
        modelo.load_state_dict(new_state_dict)
        print("✓ Modelo pre-entrenado cargado exitosamente!")
    else:
        print("⚠️  ADVERTENCIA: No se encontraron pesos pre-entrenados")
        print(f"   Buscando en: {peso_path}")
        print("   Descarga con: wget https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_gray_blind.pth -O model_zoo/dncnn_gray_blind.pth")
        print("   Continuando con pesos aleatorios (resultados serán pobres)...")
    
    modelo.eval()
    
    # 3. Preprocesar
    img_tensor = preprocesar_para_modelo(img_original)
    img_tensor = img_tensor.to(device)
    
    # 4. Inferencia
    print("Procesando imagen...")
    with torch.no_grad():
        img_denoised = modelo(img_tensor)
    
    # 5. Posprocesar
    img_limpia = posprocesar_resultado(img_denoised, img_original.shape)
    img_original_norm = normalizar_imagen(img_original)
    
    # 6. Visualizar resultados
    if mostrar_resultados:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_original_norm, cmap='gray')
        axes[0].set_title('Imagen Original')
        axes[0].axis('off')
        
        axes[1].imshow(img_limpia, cmap='gray')
        axes[1].set_title('Imagen Procesada (DnCNN)')
        axes[1].axis('off')
        
        diferencia = np.abs(img_original_norm - img_limpia)
        axes[2].imshow(diferencia, cmap='hot')
        axes[2].set_title('Diferencia (Ruido Estimado)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('resultado_denoising.png', dpi=150, bbox_inches='tight')
        print("✓ Resultado guardado como 'resultado_denoising.png'")
        plt.show()
    
    return img_limpia, img_original_norm

# ==================== EJEMPLO DE USO ====================
if __name__ == "__main__":
    import sys
    
    # Obtener ruta de imagen desde argumentos o usar default
    if len(sys.argv) > 1:
        ruta_imagen = sys.argv[1]
        print(f"📥 Imagen recibida desde Qt: {ruta_imagen}")
    else:
        # Ruta por defecto si se ejecuta directamente
        ruta_imagen = "imgAnalizar/ApicePulmonar/L143_QD_1_1.CT.0004.0053.2015.12.22.20.45.11.504991.358762830.IMA"
        print(f"📂 Usando ruta por defecto: {ruta_imagen}")
    
    # Verificar que la ruta existe
    if not os.path.exists(ruta_imagen):
        print(f"❌ Error: No se encontró la imagen en '{ruta_imagen}'")
        sys.exit(1)
    
    try:
        # Procesar imagen
        img_limpia, img_original = reducir_ruido_ct(ruta_imagen, mostrar_resultados=False)
        
        # Guardar resultado para Qt
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Convertir a 0-255 y guardar
        img_salida = (img_limpia * 255).astype(np.uint8)
        output_path = os.path.join(output_dir, "resultado_denoising.png")
        cv2.imwrite(output_path, img_salida)
        
        print(f"\n✓ Procesamiento completado exitosamente!")
        print(f"  - Imagen original: {img_original.shape}")
        print(f"  - Imagen procesada: {img_limpia.shape}")
        print(f"  - Guardada en: {output_path}")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"❌ Error durante el procesamiento: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ==================== CARGAR PESOS PRE-ENTRENADOS ====================
"""
Para usar un modelo realmente entrenado, descarga los pesos y cárgalos así:

# Descargar de: https://github.com/cszn/KAIR/releases
modelo.load_state_dict(torch.load('dncnn_gray_blind.pth', map_location=device))

Modelos disponibles:
- dncnn_15.pth: Para ruido Gaussiano sigma=15
- dncnn_25.pth: Para ruido Gaussiano sigma=25  
- dncnn_50.pth: Para ruido Gaussiano sigma=50
- dncnn_gray_blind.pth: Para ruido desconocido (RECOMENDADO para CT)
"""