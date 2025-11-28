#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/dnn.hpp>

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkGDCMImageIO.h>
#include <itkImageRegionConstIterator.h>

#include <iostream>
#include <filesystem>
#include <string>
#include "ImageProcessor.h"
#include <filesystem>

using namespace itk;
using PixelType = signed short;
using ImageType = itk::Image<PixelType, 2>;
using namespace cv;
using namespace std;
namespace fs = filesystem;
ImageProcessor::ImageProcessor() {
    
}

ImageProcessor::~ImageProcessor() {}


Mat procesarIMG16a8(const Mat& img16) {

    if (img16.empty() || img16.type() != CV_16SC1) {
        cerr << "Error: Imagen inválida para conversión" << endl;
        return Mat();
    }

    Mat img8;
    normalize(img16, img8, 0, 255, NORM_MINMAX, CV_8U);

    return img8;
}

bool ImageProcessor::loadImage(const string& filePath) {
    cout << "Loading: " << filePath << endl;
    
    // Detectar si es DICOM/IMA o imagen estándar
    string ext = fs::path(filePath).extension().string();
    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == ".ima" || ext == ".dcm") {
        // Cargar con ITK
        try {
            using ReaderType = itk::ImageFileReader<ImageType>;
            using ImageIOType = itk::GDCMImageIO;
            
            auto dicomIO = ImageIOType::New();
            auto reader = ReaderType::New();
            reader->SetFileName(filePath);
            reader->SetImageIO(dicomIO);
            reader->Update();
            
            auto itkImage = reader->GetOutput();
            auto region = itkImage->GetLargestPossibleRegion();
            auto size = region.GetSize();
            
            int width = size[0];
            int height = size[1];
            
            m_rawImage = Mat(height, width, CV_16SC1);
            
            itk::ImageRegionConstIterator<ImageType> it(itkImage, region);
            int idx = 0;
            for (it.GoToBegin(); !it.IsAtEnd(); ++it, ++idx) {
                int y = idx / width;
                int x = idx % width;
                m_rawImage.at<short>(y, x) = it.Get();
            }
            
            // Convertir a 8 bits para visualización
            normalize(m_rawImage, m_originalImage, 0, 255, NORM_MINMAX, CV_8UC1);
            
            cout << "DICOM/IMA loaded: " << width << "x" << height << endl;
            return true;
            
        } catch (const itk::ExceptionObject& ex) {
            cerr << "ITK Error: " << ex << endl;
            return false;
        }
    } else {
        // Cargar imagen estándar con OpenCV
        m_originalImage = imread(filePath, IMREAD_GRAYSCALE);
        if (m_originalImage.empty()) {
            cerr << "Error: Could not load image with OpenCV" << endl;
            return false;
        }
        
        // Convertir a 16 bits para compatibilidad
        m_originalImage.convertTo(m_rawImage, CV_16SC1);
        
        cout << "Standard image loaded: " << m_originalImage.cols 
                  << "x" << m_originalImage.rows << endl;
        //mas informacion de la imagen
        cout << "Type: " << m_originalImage.type() << endl;
        cout << "Channels: " << m_originalImage.channels() << endl;
        return true;
    }
}

Mat equializadaHistograma(const Mat& img) {
    // Mat imgEcualizada;
    // equalizeHist(img, imgEcualizada);
    // return imgEcualizada;

    
    Mat img_lab;
    if (img.empty()) {
        cout << "Error: No se pudo cargar la imagen." << endl;
        return Mat();
    }

    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(2.0);  // Límite de contraste
    clahe->setTilesGridSize(cv::Size(8, 8));  // Tamaño de grid
    
    Mat img_clahe;
    clahe->apply(img, img_clahe);
    
    
    return img_clahe;
}


Mat stretchingParaTomografia(const cv::Mat& img, int min_intensidad = 50, int max_intensidad = 250) {
    cv::Mat resultado;
    
    // Aplicar stretching en un rango específico
    double alpha = 255.0 / (max_intensidad - min_intensidad);
    double beta = -min_intensidad * alpha;
    
    img.convertTo(resultado, CV_8U, alpha, beta);
    
    // Clip valores
    cv::Mat clipped;
    cv::max(resultado, 0, clipped);
    cv::min(clipped, 255, resultado);
    
    return resultado;
}

Mat ImageProcessor::deteccionHuesos(int a, int b) {
    // A es el valor del umbral (ej. 200)
    // B es el valor máximo (normalmente 255)
    
    // 1. Validar imagen
    // Usamos clone() para no modificar la original si es una variable miembro
    // Mat ania = m_originalImage.clone(); 
    // Mat img = equializadaHistograma(ania);
    Mat img = m_originalImage.clone();
    if (img.empty()) {
        cout << "Error: No hay imagen cargada en m_originalImage." << endl;
        return Mat();
    }

    // 2. Pre-procesamiento
    Mat imgBlur;
    GaussianBlur(img, imgBlur, cv::Size(5, 5), 0);

    // 3. Umbralización
    Mat maskHuesos;
    // CORRECCIÓN 1: Usar THRESH_BINARY.
    // Necesitamos bordes definidos (0 o 255) para que findContours funcione bien.
    threshold(imgBlur, maskHuesos, (double)a, (double)b, THRESH_BINARY);

    // 4. Encontrar Contornos
    // CORRECCIÓN 2: Usar 'vector<vector<Point>>'
    vector<vector<cv::Point>> contours;
    vector<Vec4i> hierarchy;
    
    // Encontrar los bordes de las zonas blancas
    findContours(maskHuesos, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 5. Filtrar por Área (Eliminar ruido)
    Mat maskSoloHuesos = Mat::zeros(maskHuesos.size(), CV_8UC1);

    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);

        // Si el área es mayor a 50 pixeles, lo dibujamos (es hueso).
        // Si es menor, lo ignoramos (es ruido/manchas).
        if (area > 50.0) { 
            drawContours(maskSoloHuesos, contours, (int)i, Scalar(255), -1);
        }
    }

    return maskSoloHuesos;
}

Mat ImageProcessor::deteccionPulmones(int a, int b,int tamanio) {
    
    
    Mat img = m_originalImage.clone();

    if (img.empty()) {
        std::cerr << "Error: No se pudo cargar la imagen." << std::endl;
        return Mat();
    }

    // 2. Umbralización (Thresholding)
    // Convertimos lo oscuro (aire) en blanco (255) y el tejido en negro (0)
    Mat binary;
    threshold(img, binary, a, b, THRESH_BINARY_INV);

    // 3. Eliminar el aire exterior (Fondo)
    // Copiamos la imagen binaria para crear la máscara
    Mat mask = binary.clone();

    // Aplicamos FloodFill desde la esquina (0,0).
    // Rellenamos con negro (0) todo el blanco conectado al borde.
    // En C++, floodFill modifica la imagen 'mask' directamente.
    floodFill(mask, cv::Point(0, 0), Scalar(0));

    // 4. Operaciones Morfológicas (Mejora de la máscara)
    // Creamos un elemento estructurante de 3x3 (rectángulo)
    Mat kernel = getStructuringElement(MORPH_RECT, cv::Size(tamanio, tamanio));
    
    // Aplicamos "Closing" para rellenar huecos internos (vasos sanguíneos)
    // Iterations = 2 para asegurar un buen relleno
    morphologyEx(mask, mask, MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);

    // 5. Aplicar la máscara a la imagen original
    Mat resultado;
    // bitwise_and toma (src1, src2, destination, mask)
    bitwise_and(img, img, resultado, mask);

    return resultado;
}

Mat ImageProcessor::deteccionMuscular(int a, int b) {
    Mat img = m_originalImage.clone();

    if (img.empty()) {
        cout << "Error: No se pudo cargar la imagen." << endl;
        return Mat();
    }

    // 2. Suavizado (GaussianBlur)
    Mat imgBlur;
    // Size(5,5) ya no necesita cv::Size
    GaussianBlur(img, imgBlur, cv::Size(5, 5), 0);

    // 3. Rango de intensidad para Músculos
    Mat mask;
    
    // Scalar(...) define el valor del píxel
    inRange(imgBlur, Scalar(a), Scalar(b), mask);

    // 4. Limpieza (Morfología)
    // MORPH_RECT y MORPH_OPEN son constantes del namespace
    Mat kernel = getStructuringElement(MORPH_RECT, cv::Size(3, 3));
    Mat maskClean;
    morphologyEx(mask, maskClean, MORPH_OPEN, kernel);

    // 5. Aplicar máscara
    Mat result;
    img.copyTo(result, maskClean);

    return result;

}
Mat ImageProcessor::imgEcualizada() {
    Mat img = m_originalImage;
    Mat img_lab;
    if (img.empty()) {
        cout << "Error: No se pudo cargar la imagen." << endl;
        return Mat();
    }

    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(4.0);  // Límite de contraste
    clahe->setTilesGridSize(cv::Size(8, 8));  // Tamaño de grid
    
    Mat img_clahe;
    clahe->apply(img, img_clahe);
    
    
    return img_clahe;
}

Mat ImageProcessor::contrastStretching(int a, int b) {
    Mat img = m_originalImage.clone();
    if (img.empty()) {
        cout << "Error: No se pudo cargar la imagen." << endl;
        return Mat();
    }

    Mat img_stretched = stretchingParaTomografia(img, a, b);
    return img_stretched;
}


// Mat ImageProcessor::deteccionHuesos(int a, int b) {
//     Mat img = m_originalImage;

//     if (img.empty()) {
//         cout << "Error: No se pudo cargar la imagen. Revisa el nombre del archivo." << endl;
//         return Mat();
//     }

//     // 1. Pre-procesamiento: Suavizar la imagen
//     // Esto es como aplicar un pequeño desenfoque para que el color sea más uniforme.
//     Mat imgBlur;
//     GaussianBlur(img, imgBlur, cv::Size(5, 5), 0);

//     // 2. Umbralización: ¡ESTA ES LA CLAVE!
//     Mat maskHuesos;
//     // CAMBIO IMPORTANTE: Hemos subido el umbral de 200 a 230.
//     // Esto significa: "Solo marca como blanco lo que sea SUPER brillante".
//     // Si con 230 desaparecen partes de las costillas, prueba con 225.
//     // Si sigue apareciendo mancha en el centro, prueba con 235 o 240.
    
//     threshold(imgBlur, maskHuesos, a, b, THRESH_BINARY);

//     // 3. Limpieza (Operación Morfológica)
//     Mat maskLimpia;
//     // Creamos un pequeño "pincel" cuadrado de 3x3 píxeles
//     Mat kernel = getStructuringElement(MORPH_RECT, cv::Size(3, 3));
    
//     // EROSIÓN: Esto "come" los bordes de las zonas blancas.
//     // Eliminará los puntitos pequeños (ruido) y limpiará los bordes rugosos.
//     erode(maskHuesos, maskLimpia, kernel);
    
//     // (Opcional) Si ves que la erosión dejó los huesos muy finos,
//     // puedes descomentar la siguiente línea para "re-inflarlos" un poco.
//     // dilate(maskLimpia, maskLimpia, kernel);
//     return maskLimpia;
// }