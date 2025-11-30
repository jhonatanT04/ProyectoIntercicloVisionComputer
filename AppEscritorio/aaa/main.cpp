#include "ImageProcessor.h"

using namespace std;
using namespace filesystem;
using namespace cv;

ImageProcessor* processor = nullptr;

// Configuración del rango
const int MIN_HU = -1200;
const int MAX_HU = 1200;
const int TRACKBAR_RANGE = MAX_HU - MIN_HU;  // 2400

// Trackbar que va de 0 a 2400 (representa -1200 a 1200)
// int a_trackbar = 1500;  // Posición inicial: 300 HU -> 300 - (-1200) = 1500
int a_h = 10;
int b_h = 255;
int k_n = 3;
int a_p = 10;
int b_p = 255;
int tamanio_p = 3;
int a_m = 10;
int b_m = 255;
int a_clahe = 2;
// void updateInfo() {
//     int a_real = a_trackbar + MIN_HU;  // Convertir a HU real
    
//     string tejido;
//     if (a_real < -500) tejido = "Aire/Pulmon";
//     else if (a_real < 0) tejido = "Grasa";
//     else if (a_real < 100) tejido = "Tejido blando";
//     else if (a_real < 300) tejido = "Musculo";
//     else tejido = "Hueso";
    
//     cout << "\rUmbral: " << a_real << " HU (" << tejido << ")      " << flush;
// }

int main(int argc, char* argv[]) {

    string imagePath = "./L14.IMA";

    if (argc >= 2) {
        imagePath = argv[1];
    }

    processor = new ImageProcessor();

    if (!processor->loadImage(imagePath)) {
        cerr << "Error al cargar la imagen: " << imagePath << endl;
        delete processor;
        return -1;
    }

    
    // namedWindow("Hueso", WINDOW_AUTOSIZE);
    // namedWindow("Pulmones", WINDOW_AUTOSIZE);
    // namedWindow("Musculos", WINDOW_AUTOSIZE);
    namedWindow("Img parametrizada", WINDOW_AUTOSIZE);
    // namedWindow("Img Segmentada", WINDOW_AUTOSIZE);
    
    // namedWindow("overlay",WINDOW_AUTOSIZE);
    // namedWindow("overlay2",WINDOW_AUTOSIZE);
    
    // Trackbar de 0 a 2400 (representa -1200 a 1200 HU)
    // createTrackbar("Umbral HU", "Hueso", &a_trackbar, TRACKBAR_RANGE);
    // createTrackbar("Segundo parametro ", "Hueso", &b, 255);
    createTrackbar("parametro 1 ", "Img parametrizada", &a_h, 255);
    createTrackbar("parametro 2 ", "Img parametrizada", &b_h, 255);
    // createTrackbar("k ", "Img parametrizada", &k_n, 30);

    // createTrackbar("tamaño ", "Pulmones", &tamanio_p, 20);
    createTrackbar("parametro 3 ", "Img parametrizada", &a_p, 255);
    createTrackbar("parametro 4 ", "Img parametrizada", &b_p, 255);

    // createTrackbar("parametro 1 ", "Musculos", &a_m, 255);
    // createTrackbar("parametro 2 ", "Musculos", &b_m, 255);
    // // setTrackbarPos("Umbral HU", "Hueso", 1500);  // 300 HU
    // // setTrackbarPos("Segundo parametro ", "Hueso", 255);

    setTrackbarPos("parametro 1 ", "Img parametrizada", 60);
    setTrackbarPos("parametro 2 ", "Img parametrizada", 150);
    // setTrackbarPos("k ", "Img parametrizada", 3);

    
    setTrackbarPos("parametro 3 ", "Img parametrizada", 2);
    setTrackbarPos("parametro 4 ", "Img parametrizada", 60);
    // setTrackbarPos("tamaño ", "Pulmones", 3);

    // setTrackbarPos("parametro 1 ", "Musculos", 55);
    // setTrackbarPos("parametro 2 ", "Musculos", 103);

    
    
    while (true) {
        // Convertir valor del trackbar a HU real
        // int a_real = a_trackbar + MIN_HU;
        // tamanio_p = (tamanio_p % 2 == 0) ? tamanio_p + 1 : tamanio_p;
        k_n = std::max(1, k_n | 1);

        b_h = std::max(1, b_h | 1);
        
        Mat img = processor->getOriginalImage();
        // Mat imgClahe = processor->applyCLAHE(img, 3);
        // Mat filtro = processor->filterBilateral(imgClahe,9);
        // Mat imgSegmentada = processor->segmentByIntensity(filtro, a_h, b_h);
        // Mat imgLimp = processor->morphOpening(imgSegmentada,k_n);
        // imgCLAHE = processor->filterGaussian(imgCLAHE,3);
        // Mat result;
        
        // imshow("Imagen Suavizada", imgCLAHE);
        // imshow("Hueso", processor->deteccionHuesos(a_h,b_h));


        // imshow("Pulmones", processor->deteccionPulmones(a_p,b_p,tamanio_p));
        // imshow("Musculos", processor->contrastStretching(a_m,b_m)); //100-150
        // imshow("Musculos", processor->deteccionMuscular(a_m,b_m));
        // updateInfo();
        img = processor->eliminarCamilla(img);
        
        // PASO 1: Preprocesamiento - CLAHE para mejorar contraste
        Mat imgCLAHE = processor->applyCLAHE(img, 3.0);
        
        // PASO 2: Suavizado para reducir ruido
        Mat imgBlur;
        GaussianBlur(imgCLAHE, imgBlur, Size(5, 5), 0);
        
        // PASO 3: Eliminar la camilla/fondo
        Mat maskCuerpo = processor->eliminarCamilla(imgBlur);
        
        // PASO 4: Segmentación por intensidad de músculos
        // Los músculos tienen intensidad media (50-120 aproximadamente)
        Mat maskMusculos;
        inRange(imgBlur, Scalar(a_h), Scalar(b_h), maskMusculos);
        
        // PASO 5: Aplicar máscara del cuerpo para eliminar exterior
        Mat maskMusculosCuerpo;
        bitwise_and(maskMusculos, maskCuerpo, maskMusculosCuerpo);
        
        // PASO 6: Eliminar estructuras óseas (alta intensidad)
        Mat maskHuesos = processor->deteccionHuesos(180, 255);
        Mat maskHuesosInv;
        bitwise_not(maskHuesos, maskHuesosInv);
        
        // Quitar huesos de la máscara muscular
        Mat maskMusculosSinHuesos;
        bitwise_and(maskMusculosCuerpo, maskHuesosInv, maskMusculosSinHuesos);
        
        // PASO 7: Eliminar grasa subcutánea (intensidad muy baja)
        Mat maskGrasa;
        inRange(imgBlur, Scalar(a_p), Scalar(b_p), maskGrasa);
        Mat maskGrasaInv;
        bitwise_not(maskGrasa, maskGrasaInv);
        
        bitwise_and(maskMusculosSinHuesos, maskGrasaInv, maskMusculosSinHuesos);
        
        // PASO 8: Limpieza morfológica
        // Opening para eliminar ruido pequeño
        Mat maskLimpia = processor->morphOpening(maskMusculosSinHuesos, 3);
        
        // Closing para rellenar huecos internos
        maskLimpia = processor->morphClosing(maskLimpia, 5);
        
        // PASO 9: Refinamiento de bordes
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
        erode(maskLimpia, maskLimpia, kernel, Point(-1, -1), 1);
        dilate(maskLimpia, maskLimpia, kernel, Point(-1, -1), 1);
        
        // PASO 10: Crear visualización con color (opcional)
        Mat resultado = processor->createColorOverlay(img, maskLimpia, Scalar(0, 0, 255), 0.5);
        


        imshow("Img parametrizada", img);
        imshow("Img Segmentada", maskLimpia);
    
        if (waitKey(23) == 27) break;
    }
    cout << endl;
    destroyAllWindows();
    delete processor;
    return 0;
}

//================ Codigo deteccion Huesos ================

///namedWindow("Imagen Suavizada", WINDOW_AUTOSIZE);
//namedWindow("Img parametrizada", WINDOW_AUTOSIZE);
    
//namedWindow("overlay",WINDOW_AUTOSIZE);
//namedWindow("overlay2",WINDOW_AUTOSIZE);
//createTrackbar("parametro 1 ", "Img parametrizada", &a_h, 255);
//createTrackbar("parametro 2 ", "Img parametrizada", &b_h, 30);
//createTrackbar("k", "Img parametrizada", &k_n, 30);


//setTrackbarPos("parametro 1 ", "Img parametrizada", 180);
//setTrackbarPos("parametro 2 ", "Img parametrizada", 10);
//setTrackbarPos("k", "Img parametrizada", 3);
// k_n = std::max(1, k_n | 1);
// b_h = std::max(1, b_h | 1);
        
// Mat img = processor->getOriginalImage();
// Mat imgCLAHE = processor->applyCLAHE(img, 3);

// Mat imgMejoramiento = processor->segmentByIntensity(imgCLAHE, a_h, 255);
// imshow("Img parametrizada", imgMejoramiento);
// 
// Mat imgMejSuavizada = processor->filterNLMeans(imgMejoramiento);
// 
// Mat suavizada2 = processor->filterMedian(imgMejSuavizada,b_h);
// suavizada2 =  processor->morphDilation(suavizada2,k_n);
// 
// imshow("Imagen Suavizada",suavizada2 );
// imshow("overlay", processor->createColorOverlay(img, imgMejoramiento, Scalar(0,0,255),0.8));
// imshow("overlay2", processor->highlightRegion("Hueso",suavizada2, img,Scalar(255,0,255)));



// Mat imgEnhanced;
//         Ptr<CLAHE> clahe = createCLAHE();
//         clahe->setClipLimit(2.0); 
//         clahe->setTilesGridSize(cv::Size(8, 8));
//         clahe->apply(img, imgEnhanced);

//         // 2. Suavizado para reducir ruido "sal y pimienta"
//         GaussianBlur(imgEnhanced, imgEnhanced, cv::Size(3, 3), 0);

//         // 3. Definición de Máscaras por Rangos (inRange)
//         Mat maskGrasa, maskMagro;

//         // Rango para Músculo con Grasa (Tonos oscuros, ej: 20 a 60)
//         inRange(imgEnhanced, Scalar(a_h), Scalar(b_h), maskGrasa);

//         // Rango para Músculo Magro (Tonos más claros, ej: 61 a 110)
//         inRange(imgEnhanced, Scalar(a_p), Scalar(b_p), maskMagro);

//         // 4. Limpieza Morfológica (Eliminar puntos aislados)
//         Mat kernel = getStructuringElement(MORPH_ELLIPSE, cv::Size(3, 3));
//         morphologyEx(maskGrasa, maskGrasa, MORPH_OPEN, kernel);
//         morphologyEx(maskMagro, maskMagro, MORPH_OPEN, kernel);

//         // 5. Visualización: Crear superposición de colores (Overlay)
//         // Convertimos la imagen original a color para poder pintar sobre ella
//         Mat resultadoColor;
//         cvtColor(img, resultadoColor, COLOR_GRAY2BGR);
        
//         // Creamos una capa para pintar los colores
//         Mat overlay = resultadoColor.clone();

//         // Pintar Músculo Graso de VIOLETA/MORADO (BGR: 128, 0, 128)
//         overlay.setTo(Scalar(128, 0, 128), maskGrasa);

//         // Pintar Músculo Magro de ROJO (BGR: 0, 0, 255)
//         overlay.setTo(Scalar(0, 0, 255), maskMagro);

//         // 6. Mezclar con transparencia (Alpha Blending)
//         // alpha 0.4 significa 40% color, 60% imagen original
//         double alpha = 0.4; 
//         addWeighted(overlay, alpha, resultadoColor, 1.0 - alpha, 0, resultadoColor);

//         // Opcional: Dibujar contornos para resaltar bordes
//         vector<vector<cv::Point>> contoursGrasa, contoursMagro;
//         findContours(maskGrasa, contoursGrasa, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//         findContours(maskMagro, contoursMagro, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

//         // Dibuja contornos finos para mejor delimitación (sin rellenar)
//         drawContours(resultadoColor, contoursGrasa, -1, Scalar(128, 0, 128), 1);
//         drawContours(resultadoColor, contoursMagro, -1, Scalar(0, 0, 255), 1);
//         imshow("Img parametrizada", resultadoColor);