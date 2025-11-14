#include <iostream>
#include <opencv2/opencv.hpp>

// ITK
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkImageRegionConstIterator.h"


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
    if (argc < 2)
    {
        std::cerr << "Uso: " << argv[0] << " <L19.IMA>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string filename = argv[1];

    // ITK types
    using PixelType = signed short; // HU values
    constexpr unsigned int Dimension = 2;

    using ImageType = itk::Image<PixelType, Dimension>;
    using ReaderType = itk::ImageFileReader<ImageType>;

    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName(filename);

    try
    {
        reader->Update();
    }
    catch (itk::ExceptionObject &ex)
    {
        std::cerr << "Error al leer DICOM: " << ex << std::endl;
        return EXIT_FAILURE;
    }

    // Rescale intensities to 0–255 (necessary for OpenCV)
    using UCharImageType = itk::Image<unsigned char, 2>;
    using RescaleFilterType = itk::RescaleIntensityImageFilter<ImageType, UCharImageType>;

    RescaleFilterType::Pointer rescaler = RescaleFilterType::New();
    rescaler->SetOutputMinimum(0);
    rescaler->SetOutputMaximum(255);
    rescaler->SetInput(reader->GetOutput());
    rescaler->Update();

    // Convert ITK image → OpenCV Mat
    cv::Mat img = ITKImageToCVMat(rescaler->GetOutput());

    // Show image
    cv::imshow("DICOM Slice", img);
    cv::waitKey(0);

    // Save image for testing
    cv::imwrite("slice_output.png", img);

    return EXIT_SUCCESS;
}
