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
int a_p = 10;
int b_p = 255;
int tamanio_p = 3;
int a_m = 10;
int b_m = 255;
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

    string imagePath = "./L19.IMA";

    if (argc >= 2) {
        imagePath = argv[1];
    }

    processor = new ImageProcessor();

    if (!processor->loadImage(imagePath)) {
        cerr << "Error al cargar la imagen: " << imagePath << endl;
        delete processor;
        return -1;
    }

    namedWindow("Imagen", WINDOW_AUTOSIZE);
    namedWindow("Hueso", WINDOW_AUTOSIZE);
    namedWindow("Pulmones", WINDOW_AUTOSIZE);
    namedWindow("Musculos", WINDOW_AUTOSIZE);
    namedWindow("Equalizada", WINDOW_AUTOSIZE);

    // Trackbar de 0 a 2400 (representa -1200 a 1200 HU)
    // createTrackbar("Umbral HU", "Hueso", &a_trackbar, TRACKBAR_RANGE);
    // createTrackbar("Segundo parametro ", "Hueso", &b, 255);
    createTrackbar("parametro 1 ", "Hueso", &a_h, 255);
    createTrackbar("parametro 2 ", "Hueso", &b_h, 255);

    createTrackbar("tamaño ", "Pulmones", &tamanio_p, 20);
    createTrackbar("parametro 1 ", "Pulmones", &a_p, 255);
    createTrackbar("parametro 2 ", "Pulmones", &b_p, 255);

    createTrackbar("parametro 1 ", "Musculos", &a_m, 255);
    createTrackbar("parametro 2 ", "Musculos", &b_m, 255);
    // setTrackbarPos("Umbral HU", "Hueso", 1500);  // 300 HU
    // setTrackbarPos("Segundo parametro ", "Hueso", 255);

    setTrackbarPos("parametro 1 ", "Hueso", 20);
    setTrackbarPos("parametro 2 ", "Hueso", 255);

    
    setTrackbarPos("parametro 1 ", "Pulmones", 2);
    setTrackbarPos("parametro 2 ", "Pulmones", 255);
    setTrackbarPos("tamaño ", "Pulmones", 3);

    setTrackbarPos("parametro 1 ", "Musculos", 55);
    setTrackbarPos("parametro 2 ", "Musculos", 103);
    
    while (true) {
        // Convertir valor del trackbar a HU real
        // int a_real = a_trackbar + MIN_HU;
        tamanio_p = (tamanio_p % 2 == 0) ? tamanio_p + 1 : tamanio_p;
        imshow("Imagen", processor->getOriginalImage());
        imshow("Hueso", processor->deteccionHuesos(a_h,b_h));


        imshow("Pulmones", processor->deteccionPulmones(a_p,b_p,tamanio_p));
        // imshow("Musculos", processor->contrastStretching(a_m,b_m)); //100-150
        imshow("Musculos", processor->deteccionMuscular(a_m,b_m));
        // updateInfo();
        imshow("Equalizada", processor->imgEcualizada());
        if (waitKey(23) == 27) break;
    }

    cout << endl;
    destroyAllWindows();
    delete processor;
    return 0;
}