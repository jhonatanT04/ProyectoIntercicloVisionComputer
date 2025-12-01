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


Mat procesarIMG16a8(const Mat img16) {

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

Mat equializadaHistograma( Mat img) {
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


Mat stretchingParaTomografia(const Mat img, int min_intensidad = 50, int max_intensidad = 250) {
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

Mat ImageProcessor::deteccionPulmones(Mat img,int a, int b,int tamanio) {
    
    
    

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

// ============ NORMALIZACIÓN Y CONTRASTE ============
Mat ImageProcessor::normalizeImage(Mat input) {
    Mat output;
    normalize(input, output, 0, 255, NORM_MINMAX, CV_8UC1);
    return output;
}

Mat ImageProcessor::contrastStretching( Mat input) {
    return normalizeImage(input);
}

Mat ImageProcessor::applyCLAHE( Mat input, double clipLimit) {
    Mat output, gray;
    if (input.channels() > 1) cvtColor(input, gray, COLOR_BGR2GRAY);
    else gray = input.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
    
    auto clahe = createCLAHE(clipLimit, cv::Size(8, 8));
    clahe->apply(gray, output);
    return output;
}

Mat ImageProcessor::histogramEqualization( Mat input) {
    Mat output, gray;
    if (input.channels() > 1) cvtColor(input, gray, COLOR_BGR2GRAY);
    else gray = input.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
    
    equalizeHist(gray, output);
    return output;
}

// ============ THRESHOLDING ============
Mat ImageProcessor::thresholdIMG( Mat input, int threshValue) {
    Mat output, gray;
    if (input.channels() > 1) cvtColor(input, gray, COLOR_BGR2GRAY);
    else gray = input.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
    
    threshold(gray, output, threshValue, 255, THRESH_BINARY);
    return output;
}

Mat ImageProcessor::thresholdOtsu( Mat input) {
    Mat output, gray;
    if (input.channels() > 1) cvtColor(input, gray, COLOR_BGR2GRAY);
    else gray = input.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
    
    threshold(gray, output, 0, 255, THRESH_BINARY | THRESH_OTSU);
    return output;
}

Mat ImageProcessor::thresholdAdaptive( Mat input, int blockSize) {
    Mat output, gray;
    if (input.channels() > 1) cvtColor(input, gray, COLOR_BGR2GRAY);
    else gray = input.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
    
    if (blockSize % 2 == 0) blockSize++;  // Asegurar que es impar
    adaptiveThreshold(gray, output, 255, ADAPTIVE_THRESH_GAUSSIAN_C, 
                         THRESH_BINARY, blockSize, 2);
    return output;
}

// ============ OPERACIONES LÓGICAS ============
Mat ImageProcessor::applyNOT( Mat input) {
    Mat output;
    bitwise_not(input, output);
    return output;
}

Mat ImageProcessor::applyAND( Mat input1,  Mat input2) {
    Mat output;
    bitwise_and(input1, input2, output);
    return output;
}

Mat ImageProcessor::applyOR( Mat input1,  Mat input2) {
    Mat output;
    bitwise_or(input1, input2, output);
    return output;
}

Mat ImageProcessor::applyXOR( Mat input1,  Mat input2) {
    Mat output;
    bitwise_xor(input1, input2, output);
    return output;
}

// ============ DETECCIÓN DE BORDES ============
Mat ImageProcessor::edgeCanny( Mat input, int low, int high) {
    Mat output, gray;
    if (input.channels() > 1) cvtColor(input, gray, COLOR_BGR2GRAY);
    else gray = input.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
    
    GaussianBlur(gray, gray, cv::Size(5, 5), 1.4);
    Canny(gray, output, low, high);
    return output;
}

Mat ImageProcessor::edgeSobel( Mat input) {
    Mat output, gray;
    if (input.channels() > 1) cvtColor(input, gray, COLOR_BGR2GRAY);
    else gray = input.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
    
    Mat gradX, gradY;
    Sobel(gray, gradX, CV_16S, 1, 0, 3);
    Sobel(gray, gradY, CV_16S, 0, 1, 3);
    
    Mat absGradX, absGradY;
    convertScaleAbs(gradX, absGradX);
    convertScaleAbs(gradY, absGradY);
    addWeighted(absGradX, 0.5, absGradY, 0.5, 0, output);
    return output;
}

Mat ImageProcessor::edgeLaplacian( Mat input) {
    Mat output, gray;
    if (input.channels() > 1) cvtColor(input, gray, COLOR_BGR2GRAY);
    else gray = input.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
    
    Mat laplacian;
    Laplacian(gray, laplacian, CV_16S, 3);
    convertScaleAbs(laplacian, output);
    return output;
}

// ============ FILTROS DE SUAVIZADO ============
Mat ImageProcessor::filterGaussian( Mat input, int ksize) {
    if (ksize % 2 == 0) ksize++;  // Asegurar impar
    Mat output;
    GaussianBlur(input, output, cv::Size(ksize, ksize), 0);
    return output;
}

Mat ImageProcessor::filterMedian( Mat input, int ksize) {
    if (ksize % 2 == 0) ksize++;  // Asegurar impar
    Mat output;
    medianBlur(input, output, ksize);
    return output;
}

Mat ImageProcessor::filterBilateral( Mat input, int d) {
    Mat output, gray;
    if (input.depth() != CV_8U) input.convertTo(gray, CV_8UC1);
    else gray = input.clone();
    
    bilateralFilter(gray, output, d, 75, 75);
    return output;
}

Mat ImageProcessor::filterMean( Mat input, int ksize) {
    Mat output;
    blur(input, output, cv::Size(ksize, ksize));
    return output;
}

Mat ImageProcessor::filterNLMeans( Mat input) {
    Mat output, gray;
    if (input.channels() > 1) cvtColor(input, gray, COLOR_BGR2GRAY);
    else gray = input.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
    
    fastNlMeansDenoising(gray, output, 10, 7, 21);
    return output;
}

// ============ MORFOLOGÍA ============
Mat ImageProcessor::morphErosion( Mat input, int ksize) {
    Mat output;
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, cv::Size(ksize, ksize));
    erode(input, output, kernel);
    return output;
}

Mat ImageProcessor::morphDilation( Mat input, int ksize) {
    Mat output;
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, cv::Size(ksize, ksize));
    dilate(input, output, kernel);
    return output;
}

Mat ImageProcessor::morphOpening( Mat input, int ksize) {
    Mat output;
    Mat kernel = getStructuringElement(MORPH_RECT, cv::Size(ksize, ksize));
    morphologyEx(input, output, MORPH_OPEN, kernel);
    return output;
}

Mat ImageProcessor::morphClosing( Mat input, int ksize) {
    Mat output;
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, cv::Size(ksize, ksize));
    morphologyEx(input, output, MORPH_CLOSE, kernel);
    return output;
}

Mat ImageProcessor::morphGradient( Mat input, int ksize) {
    Mat output;
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, cv::Size(ksize, ksize));
    morphologyEx(input, output, MORPH_GRADIENT, kernel);
    return output;
}

Mat ImageProcessor::morphTopHat( Mat input, int ksize) {
    Mat output;
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, cv::Size(ksize, ksize));
    morphologyEx(input, output, MORPH_TOPHAT, kernel);
    return output;
}

Mat ImageProcessor::morphBlackHat( Mat input, int ksize) {
    Mat output;
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, cv::Size(ksize, ksize));
    morphologyEx(input, output, MORPH_BLACKHAT, kernel);
    return output;
}

// ============ VISUALIZACIÓN ============
Mat ImageProcessor::createColorOverlay( Mat original,  Mat mask, 
                                              Scalar color, double alpha) {
    Mat output, bgr;
    if (original.channels() == 1) cvtColor(original, bgr, COLOR_GRAY2BGR);
    else bgr = original.clone();
    
    Mat overlay = bgr.clone();
    overlay.setTo(color, mask);
    addWeighted(bgr, 1 - alpha, overlay, alpha, 0, output);
    return output;
}

Mat ImageProcessor::createHeatmap( Mat input) {
    Mat output, gray;
    if (input.channels() > 1) cvtColor(input, gray, COLOR_BGR2GRAY);
    else gray = input.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
    
    applyColorMap(gray, output, COLORMAP_JET);
    return output;
}

Mat ImageProcessor::segmentByIntensity( Mat input, int lower, int upper) {
    Mat output, gray;
    if (input.channels() > 1) cvtColor(input, gray, COLOR_BGR2GRAY);
    else gray = input.clone();
    if (gray.depth() != CV_8U) gray.convertTo(gray, CV_8UC1);
    
    inRange(gray, Scalar(lower), Scalar(upper), output);
    return output;
}

// ============ RESALTAR REGIÓN (NUEVO) ============
Mat ImageProcessor::highlightRegion(String name,Mat mask,  Mat background, Scalar color) {
    vector<vector<cv::Point>> contours;
    Mat maskCopy = mask.clone();
    findContours(maskCopy, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    Mat output;
    if (background.empty()) {
        if (mask.channels() == 1) {
            cvtColor(mask, output, COLOR_GRAY2BGR);
        } else {
            output = mask.clone();
        }
    } else {
        if (background.channels() == 1) {
            cvtColor(background, output, COLOR_GRAY2BGR);
        } else {
            output = background.clone();
        }
    }
    
    // Dibujar contornos amarillos
    drawContours(output, contours, -1, color, 2);
    
    // Dibujar rectángulos y etiquetas
    for (size_t i = 0; i < contours.size(); i++) {
        Rect boundRect = boundingRect(contours[i]);
        double area = contourArea(contours[i]);
        
        // Filtrar contornos muy pequeños
        if (area > 100) {
            rectangle(output, boundRect, Scalar(0, 255, 0), 2);
            
            std::string label = name +" : " + std::to_string((int)area);
            putText(output, label, cv::Point(boundRect.x, boundRect.y - 5),
                       FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);
        }
    }
    
    return output;
}

Mat ImageProcessor::eliminarCamilla(Mat img) {
    
    // 1. Umbralizar
    Mat binaria;
    threshold(img, binaria, 30, 255, THRESH_BINARY);
    
    // 2. Eliminar ruido con morfología
    Mat limpia = morphClosing(binaria, 10);
    
    // 3. Encontrar todos los contornos
    vector<vector<cv::Point>> contours;
    findContours(limpia.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    // 4. Encontrar el contorno más grande (el cuerpo)
    if (contours.empty()) {
        cout << "No se encontraron contornos" << endl;
        return img;
    }
    
    int indiceMax = 0;
    double areaMax = 0;
    
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > areaMax) {
            areaMax = area;
            indiceMax = i;
        }
    }
    
    // 5. Crear máscara solo con el contorno del cuerpo
    Mat maskCuerpo = Mat::zeros(img.size(), CV_8U);
    drawContours(maskCuerpo, contours, indiceMax, Scalar(255), -1);
    
    // 6. Erosionar un poco para eliminar piel/grasa superficial
    Mat maskErosionada = morphErosion(maskCuerpo, 10);
    
    // 7. Aplicar máscara
    Mat resultado = Mat::zeros(img.size(), CV_8U);
    img.copyTo(resultado, maskErosionada);
    
    return resultado;
}

Mat ImageProcessor::createMultiColorOverlay(const Mat& imgOriginal, 
                                            const Mat& mask1, 
                                            const Mat& mask2, 
                                            const Mat& mask3,
                                            const Scalar& color1,
                                            const Scalar& color2,
                                            const Scalar& color3,
                                            double alpha)
{
    // Verificar que la imagen original no esté vacía
    if (imgOriginal.empty()) {
        return Mat();
    }
    
    // Convertir imagen original a BGR si es escala de grises
    Mat resultado;
    if (imgOriginal.channels() == 1) {
        cvtColor(imgOriginal, resultado, COLOR_GRAY2BGR);
    } else {
        resultado = imgOriginal.clone();
    }
    
    // Aplicar primera máscara (por ejemplo: huesos en azul)
    if (!mask1.empty()) {
        Mat overlay1 = resultado.clone();
        overlay1.setTo(color1, mask1);
        addWeighted(resultado, 1.0 - alpha, overlay1, alpha, 0, resultado);
    }
    
    // Aplicar segunda máscara (por ejemplo: pulmones en amarillo)
    if (!mask2.empty()) {
        Mat overlay2 = resultado.clone();
        overlay2.setTo(color2, mask2);
        addWeighted(resultado, 1.0 - alpha, overlay2, alpha, 0, resultado);
    }
    
    // Aplicar tercera máscara (por ejemplo: músculos en verde)
    if (!mask3.empty()) {
        Mat overlay3 = resultado.clone();
        overlay3.setTo(color3, mask3);
        addWeighted(resultado, 1.0 - alpha, overlay3, alpha, 0, resultado);
    }
    
    return resultado;
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
//     GaussianBlur(img, imgBlur, Size(5, 5), 0);

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