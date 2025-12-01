#pragma once

#include <opencv2/opencv.hpp>
#include <string>

// ITK
#include <itkImage.h>
using namespace cv;
class ImageProcessor {
public:
    using PixelType = signed short;
    using ImageType = itk::Image<PixelType, 2>;

    ImageProcessor();
    ~ImageProcessor();

    bool loadImage(const std::string& filePath);
    Mat deteccionHuesos(int a, int b);
    std::vector<Mat> deteccionPulmones(Mat img,int a, int b, int tamanio);
    Mat deteccionMuscular(int a, int b);
    Mat imgEcualizada();
    Mat contrastStretching(int a, int b);

        // ================ NORMALIZACIÓN Y CONTRASTE ========
    Mat normalizeImage(Mat input);
    Mat contrastStretching( Mat input);
    Mat applyCLAHE( Mat input, double clipLimit = 2.0);
    Mat histogramEqualization( Mat input);

    // ================ THRESHOLDING =====================
    Mat thresholdIMG( Mat input, int threshValue);
    Mat thresholdOtsu( Mat input);
    Mat thresholdAdaptive( Mat input, int blockSize = 11);

    // ================ OPERADORES LÓGICOS ===============
    Mat applyNOT( Mat input);
    Mat applyAND( Mat input1,  Mat input2);
    Mat applyOR( Mat input1,  Mat input2);
    Mat applyXOR( Mat input1,  Mat input2);

    // ================ DETECCIÓN DE BORDES ==============
    Mat edgeCanny( Mat input, int low = 50, int high = 150);
    Mat edgeSobel( Mat input);
    Mat edgeLaplacian( Mat input);

    // ================ FILTROS DE SUAVIZADO =============
    Mat filterGaussian( Mat input, int ksize = 5);
    Mat filterMedian( Mat input, int ksize = 5);
    Mat filterBilateral( Mat input, int d = 9);
    Mat filterMean( Mat input, int ksize = 5);
    Mat filterNLMeans( Mat input);

    // ================ MORFOLOGÍA =======================
    Mat morphErosion( Mat input, int ksize = 5);
    Mat morphDilation( Mat input, int ksize = 5);
    Mat morphOpening( Mat input, int ksize = 5);
    Mat morphClosing( Mat input, int ksize = 5);
    Mat morphGradient( Mat input, int ksize = 5);
    Mat morphTopHat( Mat input, int ksize = 15);
    Mat morphBlackHat( Mat input, int ksize = 15);

    // ================ SEGMENTACIÓN =====================
    Mat segmentByIntensity( Mat input, int lower, int upper);

     // ================ VISUALIZACIÓN ====================
    Mat createColorOverlay( Mat original,  Mat mask,
                               Scalar color, double alpha = 0.5);
    Mat createHeatmap( Mat input);
    
    // ⭐ NUEVO: Resaltar región con contornos, áreas y rectángulos
    Mat highlightRegion(String name, Mat mask,  Mat background,Scalar color);
    
    // ⭐ NUEVO: Extraer huesos usando umbral de Hounsfield
    Mat extractBones( Mat input, int minHU = 200, int maxHU = 3000);
    
    // ⭐ NUEVO: Overlay multicolor por rangos de intensidad
    Mat createMultiColorOverlay( Mat original);
    
    // ⭐ NUEVO: Combinar imagen con bordes
    Mat combineWithEdges( Mat input,  Mat edges, 
                             Scalar edgeColor = Scalar(0, 255, 0));

    Mat eliminarCamilla(Mat img);
    Mat createMultiColorOverlay(const Mat& imgOriginal, 
                                            const Mat& mask1, 
                                            const Mat& mask2, 
                                            const Mat& mask3,
                                            const Scalar& color1,
                                            const Scalar& color2,
                                            const Scalar& color3,
                                            double alpha);


    // Opcionalmente puedes agregar getters
    const Mat& getOriginalImage() const { return m_originalImage; }
    const Mat& getRawImage() const { return m_rawImage; }

    void analyzeRawImage();


private:
    Mat m_originalImage;  // Imagen lista para visualizar (8 bits)
    Mat m_rawImage;       // Imagen cruda cargada (16 bits)
};

