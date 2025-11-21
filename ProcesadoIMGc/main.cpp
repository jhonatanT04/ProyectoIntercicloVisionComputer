#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>   

// ITK
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkImageRegionConstIterator.h"

#include "itkGDCMImageIO.h"

namespace fs = std::filesystem;

cv::Mat ITKImageToCVMat(itk::Image<unsigned char, 2>::Pointer image)
{
    int width = image->GetLargestPossibleRegion().GetSize()[0];
    int height = image->GetLargestPossibleRegion().GetSize()[1];

    cv::Mat mat(height, width, CV_8UC1);

    itk::ImageRegionConstIterator<itk::Image<unsigned char, 2>> it(
        image, image->GetLargestPossibleRegion());

    for (int y = 0; y < height; y++)
    {
        uchar* rowPtr = mat.ptr<uchar>(y);
        for (int x = 0; x < width; x++)
        {
            rowPtr[x] = it.Get();
            ++it;
        }
    }

    return mat;
}

int main(int argc, char *argv[])
{
    std::string imagesFolder = "images";

    if (!fs::exists(imagesFolder))
    {
        std::cerr << "La carpeta 'images/' no existe. Créala junto al ejecutable." << std::endl;
        return EXIT_FAILURE;
    }

    // Buscar el primer archivo .IMA
    std::string filename;

    for (const auto& entry : fs::directory_iterator(imagesFolder))
    {
        if (entry.path().extension() == ".IMA")
        {
            filename = entry.path().string();
            break;
        }
    }

    if (filename.empty())
    {
        std::cerr << " No se encontró ninguna imagen .IMA en /images." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << " Cargando archivo: " << filename << std::endl;

    // ITK types
    using PixelType = signed short; 
    constexpr unsigned int Dimension = 2;

    using ImageType = itk::Image<PixelType, Dimension>;
    using ReaderType = itk::ImageFileReader<ImageType>;

    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName(filename);

    // --- ACTIVAR LECTOR GDCM (NECESARIO PARA IMÁGENES .IMA) ---
    itk::GDCMImageIO::Pointer gdcmIO = itk::GDCMImageIO::New();
    reader->SetImageIO(gdcmIO);

    try
    {
        reader->Update();
    }
    catch (itk::ExceptionObject &ex)
    {
        std::cerr << "Error al leer imagen IMA con GDCM: " << ex << std::endl;
        return EXIT_FAILURE;
    }

    // Rescale intensities to 0–255
    using UCharImageType = itk::Image<unsigned char, 2>;
    using RescaleFilterType = itk::RescaleIntensityImageFilter<ImageType, UCharImageType>;

    RescaleFilterType::Pointer rescaler = RescaleFilterType::New();
    rescaler->SetOutputMinimum(0);
    rescaler->SetOutputMaximum(255);
    rescaler->SetInput(reader->GetOutput());
    rescaler->Update();

    // Convertir ITK → OpenCV
    cv::Mat img = ITKImageToCVMat(rescaler->GetOutput());

    // Mostrar
    cv::imshow("Imagen IMA", img);
    cv::waitKey(0);

    // Guardar salida
    cv::imwrite("output_slice.png", img);

    std::cout << "Imagen convertida y guardada como output_slice.png" << std::endl;

    return EXIT_SUCCESS;
}
