#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QImage>
#include <QPixmap>
#include <QFileDialog>
#include <QMessageBox>
#include <QSlider>
#include <QProcess>           // ← FALTABA: Para ejecutar Python
#include <QProgressDialog>    // ← FALTABA: Para barra de progreso
#include <QFile>              // ← FALTABA: Para eliminar temp_input.png
#include <QDir>               // ← FALTABA: Para crear carpeta output
#include <QDebug>             // ← FALTABA: Para qDebug()
#include <QListWidgetItem>    // ← FALTABA: Si usas listWidget
#include <QTimer>
#include <QMutex>
#include <QStringList>
#include <opencv2/opencv.hpp>
#include "ProcesadoIMGc/CTProcessorSimple.h"
#include "aaa/ImageProcessor.h"
#include <QCloseEvent>

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
    void on_pushButton_2_clicked();     // Pipeline completo
    void on_pushButton_3_clicked();     // Aplicar Red Neuronal DnCNN
    void updateImageByCheckbox();
    void logDataToCSV(const QString &filename);
    void updateFilters();               // Actualizar sliders en tiempo real
    double getCurrentRAMUsageMB();

protected:
    void closeEvent(QCloseEvent *event) override;

private:
    // Muestreo de RAM en tiempo real
    QTimer *ramSampleTimer = nullptr;    // Timer de muestreo (p.ej. 1 ms)
    QTimer *csvFlushTimer = nullptr;     // Timer para vaciar buffer al disco
    QStringList ramBuffer;               // Buffer en memoria de líneas CSV
    QMutex bufferMutex;                  // Protege el buffer
    std::ofstream csvFile;               // Archivo abierto durante la sesión
    QString csvFilePath;                 // Ruta del CSV abierto

    void startRamSampling(int intervalMs = 1);
    void stopRamSampling();
    void flushBufferToFile();

private:
    Ui::MainWindow *ui;
    CTImageProcessor *processor;        // Procesador de imágenes
    ImageProcessor *procesado;          // Procesador de imágenes
    cv::Mat currentImage;               // Imagen original
    cv::Mat pipelineImage;              // Imagen con pipeline interno
    cv::Mat dncnnImage;   // Imagen suavizada con DnCNN
    bool dncnnApplied = false;
    cv::Mat originalImage;
    void showImage(const cv::Mat &img);
    QImage matToQImage(const cv::Mat &mat);
    // void applyInternalPipeline();       // Aplica filtros internos automáticamente
};

#endif // MAINWINDOW_Hy