#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRGBToLuminanceImageFilter.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    

    if (argc < 3) {
        std::cerr << "Uso: " << argv[0] << " <entrada.jpg> <salida.jpg>" << std::endl;
        return EXIT_FAILURE;
    }
    
    const char* inputFileName = argv[1];
    const char* outputFileName = argv[2];
    Mat imagen = imread(inputFileName);
    namedWindow("ventana",WINDOW_AUTOSIZE);

    imshow("ventana",imagen);

    waitKey(0);
    destroyAllWindows();
    try {
        
        using RGBPixelType = itk::RGBPixel<unsigned char>;
        using RGBImageType = itk::Image<RGBPixelType, 2>;
        using GrayImageType = itk::Image<unsigned char, 2>;

        // Crear lector
        using ReaderType = itk::ImageFileReader<RGBImageType>;
        ReaderType::Pointer reader = ReaderType::New();
        reader->SetFileName(inputFileName);

        // Convertir a escala de grises
        using FilterType = itk::RGBToLuminanceImageFilter<RGBImageType, GrayImageType>;
        FilterType::Pointer rgbToGray = FilterType::New();
        rgbToGray->SetInput(reader->GetOutput());

        // Crear escritor
        using WriterType = itk::ImageFileWriter<GrayImageType>;
        WriterType::Pointer writer = WriterType::New();
        writer->SetFileName(outputFileName);
        writer->SetInput(rgbToGray->GetOutput());

        // Ejecutar el pipeline
        writer->Update();

        cout << "✅ Imagen convertida exitosamente : " << outputFileName << endl;
    }
    catch (itk::ExceptionObject & error) {
        cerr << "❌ Error procesando la imagen: " << error << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
