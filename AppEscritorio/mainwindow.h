#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QImage>
#include <QPixmap>
#include <QFileDialog>
#include <QMessageBox>
#include <QSlider>
#include <opencv2/opencv.hpp>
#include "ProcesadoIMGc/CTProcessorSimple.h"
#include "aaa/ImageProcessor.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_pushButton_clicked();       // Cargar Imagen
    void on_pushButton_2_clicked();     // Pipeline interno (botón opcional)
    void on_pushButton_3_clicked();     // Aplicar Red

    void updateFilters();               // Actualizar sliders en tiempo real

private:
    Ui::MainWindow *ui;
    CTImageProcessor *processor;        // Procesador de imágenes
    ImageProcessor *procesado;           // Procesador de imágenes
    cv::Mat currentImage;               // Imagen original
    cv::Mat pipelineImage;              // Imagen con pipeline interno

    QImage matToQImage(const cv::Mat &mat);

    void applyInternalPipeline();       // Aplica filtros internos automáticamente
};

#endif // MAINWINDOW_H
