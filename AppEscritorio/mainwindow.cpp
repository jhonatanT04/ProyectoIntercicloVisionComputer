#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "pipelinedialog.h"
#include <QTimer>
#include <fstream>
#include <unistd.h>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    processor = new CTImageProcessor("output");  
    procesado = new ImageProcessor();

    QTimer *ramTimer = new QTimer(this);
    connect(ramTimer, &QTimer::timeout, this, [this]() {
        double ramMB = getCurrentRAMUsageMB();
        ui->label_RAM->setText(QString("RAM usada: %1 MB").arg(ramMB, 0, 'f', 1));
    });

    ramTimer->start(1000);   // actualizar cada 1 segundo
    // Configurar rangos de sliders
    ui->horizontalSlider->setRange(100, 255);      
    ui->horizontalSlider->setValue(170);

    ui->horizontalSlider_2->setRange(0, 30);    
    ui->horizontalSlider_2->setValue(3);

    ui->horizontalSlider_3->setRange(0, 30);   
    ui->horizontalSlider_3->setValue(3);


    ui->horizontalSlider_4->setRange(30, 255);   
    ui->horizontalSlider_4->setValue(70);

    ui->horizontalSlider_5->setRange(1, 255);   
    ui->horizontalSlider_5->setValue(200);

    

    ui->horizontalSlider_7->setRange(30, 255);   
    ui->horizontalSlider_7->setValue(30);

    ui->horizontalSlider_8->setRange(1, 255);   
    ui->horizontalSlider_8->setValue(200);

    ui->horizontalSlider_9->setRange(0, 30);    
    ui->horizontalSlider_9->setValue(5);
    
    // Conectar sliders para tiempo real
    connect(ui->horizontalSlider, &QSlider::valueChanged,
            this, &MainWindow::updateFilters);
    connect(ui->horizontalSlider_2, &QSlider::valueChanged,
            this, &MainWindow::updateFilters);
    connect(ui->horizontalSlider_3, &QSlider::valueChanged,
            this, &MainWindow::updateFilters);
    connect(ui->horizontalSlider_4, &QSlider::valueChanged,
            this, &MainWindow::updateFilters);
    connect(ui->horizontalSlider_5, &QSlider::valueChanged,
            this, &MainWindow::updateFilters);
    
    connect(ui->horizontalSlider_7, &QSlider::valueChanged,
            this, &MainWindow::updateFilters);
    connect(ui->horizontalSlider_8, &QSlider::valueChanged,
            this, &MainWindow::updateFilters);
    connect(ui->horizontalSlider_9, &QSlider::valueChanged,
            this, &MainWindow::updateFilters);
    connect(ui->checkBox_dncnn_suavizado, &QCheckBox::stateChanged,
        this, &MainWindow::updateImageByCheckbox);

}

MainWindow::~MainWindow()
{
    delete processor;
    delete ui;
    delete procesado;
}

// ==================== Conversión Mat -> QImage ====================
QImage MainWindow::matToQImage(const cv::Mat &mat)
{
    if(mat.empty()) return QImage();
    
    if(mat.type() == CV_8UC1) {
        return QImage(mat.data, mat.cols, mat.rows, mat.step,
                      QImage::Format_Grayscale8).copy();
    }
    if(mat.type() == CV_8UC3) {
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
        return QImage(rgb.data, rgb.cols, rgb.rows, rgb.step,
                      QImage::Format_RGB888).copy();
    }
    
    // Si es otro tipo, normalizar primero
    cv::Mat normalized;
    cv::normalize(mat, normalized, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    return QImage(normalized.data, normalized.cols, normalized.rows, 
                  normalized.step, QImage::Format_Grayscale8).copy();
}

// ==================== BOTÓN 1: Cargar Imagen ====================
void MainWindow::on_pushButton_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(
           this, "Seleccionar imagen CT", "",
           "Imagenes (*.png *.jpg *.jpeg *.bmp *.IMA *.dcm)");

    //QString fileName = "/home/jhonatan/VisualCodeStudio/ProyectoIntercicloVisionComputer/AppEscritorio/aaa/build/L19.IMA";
    
    if(fileName.isEmpty()) return;

    if (!processor->loadImage(fileName.toStdString())) {
        QMessageBox::warning(this, "Error", "No se pudo cargar la imagen.");
        return;
    }
    
    procesado->loadImage(fileName.toStdString());
    Mat img = procesado->getOriginalImage();
    currentImage = img.clone();  

    if (ui->checkBox_dncnn_suavizado->isChecked()) {

        // Ejecutar la función del botón 3 que usa DnCNN
        on_pushButton_3_clicked();

        // Guardar imagen obtenida por DnCNN
        dncnnApplied = true;
        currentImage = dncnnImage.clone();   // USAR imagen DnCNN para todo

        // Mostrar en label para verificar
        ui->label->setPixmap(QPixmap::fromImage(matToQImage(dncnnImage)
                          .scaled(ui->label->width(), ui->label->height(),
                                  Qt::KeepAspectRatio, Qt::SmoothTransformation)));
    } 
    else {
        dncnnApplied = false;
    }

    originalImage = img.clone();
    currentImage = img.clone();

    showImage(originalImage);
    updateImageByCheckbox();

    // Aplicar pipeline interno automáticamente
    // applyInternalPipeline();

    // Actualizar sliders en tiempo real
    updateFilters();
}

// ==================== Pipeline Interno OPTIMIZADO ====================


// ==================== Update Filters (Tiempo Real) ====================
void MainWindow::updateFilters()
{
    if(currentImage.empty()) return;

    // ============================================================
    // OBTENER VALORES DE SLIDERS
    // ============================================================
    int a_h = ui->horizontalSlider->value();
    int b_h = std::max(1, ui->horizontalSlider_2->value() | 1);
    int k_n = std::max(1, ui->horizontalSlider_3->value() | 1);
    
    int a_m = ui->horizontalSlider_4->value();
    int b_m = ui->horizontalSlider_5->value();
    


    int a_p = ui->horizontalSlider_7->value();
    int b_p = ui->horizontalSlider_8->value();
    int c_p = std::max(1, ui->horizontalSlider_9->value() | 1);
    // ============================================================
    // PROCESAMIENTO
    // ============================================================
    Mat img = procesado->getOriginalImage();

    Mat imgSuavizada = procesado->filterMedian(img, k_n);
    //Mat imgSuavizada = procesado->filterMedian(img, 5);
    
    
    // Mat img = procesado->eliminarCamilla(img);
    bool checkedDnCNN = ui->checkBox_dncnn_suavizado->isChecked();
    if (checkedDnCNN && dncnnApplied) {
        img = dncnnImage.clone();   // Usar imagen suavizada
    }
    else{
        img = imgSuavizada.clone();
    }

    Mat imgCLAHE = processor->applyCLAHE( procesado->eliminarCamilla(img), 3);
    Mat imgMejoramiento = processor->segmentByIntensity(imgCLAHE, a_h, 255);

    Mat imgMejSuavizada = processor->filterNLMeans(imgMejoramiento);
    Mat suavizada2 = processor->filterMedian(imgMejSuavizada, b_h);

    //suavizada2 = processor->morphDilation(suavizada2, k_n);

    std::vector<Mat> capasPulmones = procesado->deteccionPulmones(procesado->eliminarCamilla(img), a_p,b_p,c_p);
    
    Mat pulmones = capasPulmones[2];

    // PASO 1: Preprocesamiento - CLAHE para mejorar contraste
    Mat imgCLAHEmus = procesado->applyCLAHE(procesado->eliminarCamilla(img), 3.0);
    
    // PASO 2: Suavizado para reducir ruido
    Mat imgBlur;
    GaussianBlur(imgCLAHEmus, imgBlur, Size(5, 5), 0);
    
    // PASO 3: Eliminar la camilla/fondo
    Mat maskCuerpo = imgBlur;
    
    // PASO 4: Segmentación por intensidad de músculos
    // Los músculos tienen intensidad media (50-120 aproximadamente)
    Mat maskMusculos;
    inRange(imgBlur, Scalar(a_m), Scalar(b_m), maskMusculos);
    
    // PASO 5: Aplicar máscara del cuerpo para eliminar exterior
    Mat maskMusculosCuerpo;
    bitwise_and(maskMusculos, maskCuerpo, maskMusculosCuerpo);
    
    // PASO 6: Eliminar estructuras óseas (alta intensidad)
    // Mat maskHuesos = procesado->filterMedian(imgMejoramiento,k_n);
    Mat maskHuesosInv;
    bitwise_not(imgMejoramiento, maskHuesosInv);
    
    // Quitar huesos de la máscara muscular
    Mat maskMusculosSinHuesos;
    bitwise_and(maskMusculosCuerpo, maskHuesosInv, maskMusculosSinHuesos);
    
    // PASO 7: Eliminar grasa subcutánea (intensidad muy baja)
    Mat maskGrasa;
    inRange(imgBlur, Scalar(12), Scalar(12), maskGrasa);
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
    Mat resultado = processor->createColorOverlay(procesado->getOriginalImage(), maskLimpia, Scalar(0, 255, 255), 0.6);
    
    Mat resultadoMultiColor = procesado->createMultiColorOverlay(
        procesado->getOriginalImage(),                              // Imagen original
        suavizada2,                       // Máscara 1: Huesos
        pulmones,                         // Máscara 2: Pulmones
        maskLimpia,                       // Máscara 3: Músculos
        Scalar(255, 0, 0),               // Azul para huesos (BGR)
        Scalar(0, 155, 0),             // Amarillo para pulmones (BGR)
        Scalar(0, 255, 255),               // Verde para músculos (BGR)
        0.6                              // Transparencia
    );

    ui->label->setPixmap(QPixmap::fromImage(matToQImage(procesado->createColorOverlay(procesado->getOriginalImage(), suavizada2, Scalar(255, 0, 0), 0.6))
                        .scaled(ui->label->width(), ui->label->height(), 
                               Qt::KeepAspectRatio, Qt::SmoothTransformation)));

    ui->label_2->setPixmap(QPixmap::fromImage(matToQImage(procesado->createColorOverlay(procesado->getOriginalImage(), pulmones, Scalar(0, 155, 0), 0.6))
                          .scaled(ui->label_2->width(), ui->label_2->height(), 
                                 Qt::KeepAspectRatio, Qt::SmoothTransformation)));

    ui->label_3->setPixmap(QPixmap::fromImage(matToQImage(resultado)
                          .scaled(ui->label_3->width(), ui->label_3->height(), 
                                 Qt::KeepAspectRatio, Qt::SmoothTransformation)));
    
    ui->label_5->setPixmap(QPixmap::fromImage(matToQImage(imgSuavizada)
                          .scaled(ui->label_5->width(), ui->label_5->height(), 
                                 Qt::KeepAspectRatio, Qt::SmoothTransformation)));

    ui->label_6->setPixmap(QPixmap::fromImage(matToQImage(resultadoMultiColor)
                          .scaled(ui->label_6->width(), ui->label_6->height(), 
                                 Qt::KeepAspectRatio, Qt::SmoothTransformation)));
                                 
}

// ==================== BOTÓN 2: Pipeline Completo ====================
void MainWindow::on_pushButton_2_clicked()
{
    if(currentImage.empty()) {
        QMessageBox::warning(this, "Error", "Primero cargue una imagen.");
        return;
    }
    std::vector<std::pair<QString, cv::Mat>> stages;

    Mat img = procesado->getOriginalImage();

    stages.push_back({"Imagen original", img});

    bool checkedDnCNN = ui->checkBox_dncnn_suavizado->isChecked();

    if (checkedDnCNN && dncnnApplied) {
        stages.push_back({"Imagen Suaviazada DnCNN", dncnnImage.clone()});
    }
    
    
    int a_h = ui->horizontalSlider->value();
    int b_h = std::max(1, ui->horizontalSlider_2->value() | 1);
    int k_n = std::max(1, ui->horizontalSlider_3->value() | 1);
    
    int a_m = ui->horizontalSlider_4->value();
    int b_m = ui->horizontalSlider_5->value();
    


    int a_p = ui->horizontalSlider_7->value();
    int b_p = ui->horizontalSlider_8->value();
    int c_p = std::max(1, ui->horizontalSlider_9->value() | 1);


    Mat imgSuavizada = procesado->filterMedian(img, k_n);
    stages.push_back({"Imagen suavizada Filtro mediana", imgSuavizada});
    
    Mat binariaCamilla;
    threshold(img, binariaCamilla, 30, 255, THRESH_BINARY);
    stages.push_back({"Umbralizacion (Camilla)", binariaCamilla});

    Mat limpia = procesado->morphClosing(binariaCamilla, 10);
    stages.push_back({"Eliminar ruido", procesado->morphClosing(limpia, 10)});

    std::vector<std::vector<cv::Point>> contours;
    findContours(limpia.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        std::cout << "No se encontraron contornos" << std::endl;
        return ;
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

    Mat maskCuerpo = Mat::zeros(img.size(), CV_8U);
    drawContours(maskCuerpo, contours, indiceMax, Scalar(255), -1);
    stages.push_back({"Contornos Detectados", maskCuerpo});

    Mat maskErosionada = procesado->morphErosion(maskCuerpo, 10);
    Mat imgSinCamilla = Mat::zeros(img.size(), CV_8U);
    img.copyTo(imgSinCamilla, maskErosionada);

    stages.push_back({"Imagen sin la camilla", imgSinCamilla});


    Mat imgCLAHE = processor->applyCLAHE(imgSinCamilla,3);
    stages.push_back({"Ecualizacion del Histograma", imgSinCamilla});

    Mat imgHuesosSegmentada =  processor->segmentByIntensity(imgCLAHE, a_h, 255);
    stages.push_back({"Ecualizacion del Histograma", imgHuesosSegmentada});

    Mat imgMejSuavizadaHuesos = processor->filterNLMeans(imgHuesosSegmentada);
    Mat segmentoHuesos = processor->filterMedian(imgMejSuavizadaHuesos, b_h);

    stages.push_back({"Resultado de huesos", segmentoHuesos});

    std::vector<Mat> capasPulmones = procesado->deteccionPulmones(procesado->eliminarCamilla(img), a_p,b_p,c_p);

    stages.push_back({"Imagen binaria (Pulmones)", capasPulmones[0]});
    stages.push_back({"Closthig", capasPulmones[1]});
    stages.push_back({"Resultado de Pulmones", capasPulmones[2]});


    Mat imgBlur;
    GaussianBlur(imgCLAHE, imgBlur, Size(5, 5), 0);
    Mat maskMusculos;
    inRange(imgBlur, Scalar(a_m), Scalar(b_m), maskMusculos);

    stages.push_back({"Mascara binaria (Musculos)", maskMusculos});

    Mat maskMusculosCuerpo;
    bitwise_and(maskMusculos, imgBlur, maskMusculosCuerpo);
    
    Mat maskHuesos = procesado->filterMedian(segmentoHuesos,k_n);
    
    Mat maskHuesosInv;
    bitwise_not(maskHuesos, maskHuesosInv);
    
    Mat maskMusculosSinHuesos;
    bitwise_and(maskMusculosCuerpo, maskHuesosInv, maskMusculosSinHuesos);

    Mat maskGrasa;
    inRange(imgBlur, Scalar(12), Scalar(12), maskGrasa);
    Mat maskGrasaInv;
    bitwise_not(maskGrasa, maskGrasaInv);
    
    bitwise_and(maskMusculosSinHuesos, maskGrasaInv, maskMusculosSinHuesos);

    Mat maskDeteccionMusculos = processor->morphOpening(maskMusculosSinHuesos, 3);
    
    // Closing para rellenar huecos internos
    maskDeteccionMusculos = processor->morphClosing(maskDeteccionMusculos, 5);

    stages.push_back({"Mascara limpia ", maskDeteccionMusculos});
    // PASO 9: Refinamiento de bordes

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    erode(maskDeteccionMusculos, maskDeteccionMusculos, kernel, Point(-1, -1), 1);
    dilate(maskDeteccionMusculos, maskDeteccionMusculos, kernel, Point(-1, -1), 1);

    stages.push_back({"Refinamiento de bordes", maskDeteccionMusculos});

    stages.push_back({"Segmentacion de Huesos", procesado->createColorOverlay(procesado->getOriginalImage(), imgHuesosSegmentada, Scalar(255, 0, 0), 0.6)});
    stages.push_back({"Segmentacion de Musculos", procesado->createColorOverlay(procesado->getOriginalImage(), maskDeteccionMusculos, Scalar(0, 155, 0), 0.6)});
    stages.push_back({"Segmentacion de Pulmones", procesado->createColorOverlay(procesado->getOriginalImage(), capasPulmones[2], Scalar(0, 255, 255), 0.6)});

    PipelineDialog dlg(stages, this);
    dlg.exec();
}

// ==================== BOTÓN 3: Red Neuronal DnCNN ====================
void MainWindow::on_pushButton_3_clicked()
{
    if(currentImage.empty()) {
        QMessageBox::warning(this, "Error", "Cargue una imagen primero.");
        return;
    }

    // 1. Guardar imagen temporal
    QString tempImagePath = "temp_input.png";
    bool saved = cv::imwrite(tempImagePath.toStdString(), currentImage);
    
    if(!saved) {
        QMessageBox::critical(this, "Error", "No se pudo guardar la imagen temporal.");
        return;
    }
    
    // 2. Crear directorio de salida
    QDir outputDir("output");
    if(!outputDir.exists()) {
        outputDir.mkpath(".");
    }
    
    // 3. Preparar proceso Python
    QProcess process;
    QString pythonScript = "/ania/main.py";
    QString outputPath = "output/resultado_denoising.png";
    
    // 4. Mostrar barra de progreso
    QProgressDialog progress("Aplicando DnCNN (modelo pre-entrenado)...", 
                            "Cancelar", 0, 0, this);
    progress.setWindowModality(Qt::WindowModal);
    progress.setMinimumDuration(0);
    progress.setValue(0);
    progress.show();
    QApplication::processEvents();
    
    // 5. Ejecutar Python
    QStringList arguments;
    arguments << pythonScript << tempImagePath;
    
    qDebug() << "Ejecutando:" << "python3" << arguments;
    
    process.start("python3", arguments);
    
    // Esperar hasta 60 segundos
    bool finished = process.waitForFinished(60000);
    
    progress.close();
    
    if(!finished) {
        QMessageBox::critical(this, "Error", 
            "El proceso Python tardó demasiado o falló.\n" + 
            process.errorString());
        QFile::remove(tempImagePath);
        return;
    }
    
    // 6. Verificar código de salida
    int exitCode = process.exitCode();
    QString stdOutput = process.readAllStandardOutput();
    QString stdError = process.readAllStandardError();
    
    qDebug() << "Exit code:" << exitCode;
    qDebug() << "Output:" << stdOutput;
    
    if(exitCode != 0) {
        QMessageBox::critical(this, "Error de Python", 
            "El script falló con código " + QString::number(exitCode) + ":\n\n" + 
            stdError + "\n\n" + stdOutput);
        QFile::remove(tempImagePath);
        return;
    }
    
    // 7. Verificar archivo de salida
    if(!QFile::exists(outputPath)) {
        QMessageBox::critical(this, "Error", 
            "No se generó el archivo esperado:\n" + outputPath);
        QFile::remove(tempImagePath);
        return;
    }
    
    // 8. Cargar imagen procesada por DnCNN
    cv::Mat denoised = cv::imread(outputPath.toStdString(), cv::IMREAD_UNCHANGED);
    
    if(denoised.empty()) {
        QMessageBox::critical(this, "Error", 
            "No se pudo cargar la imagen procesada.");
        QFile::remove(tempImagePath);
        return;
    }
    
    // 9. Convertir a formato adecuado
    cv::Mat denoisedDisplay;
    if(denoised.channels() == 1) {
        denoisedDisplay = denoised.clone();
    } else if(denoised.channels() == 3) {
        cv::cvtColor(denoised, denoisedDisplay, cv::COLOR_BGR2GRAY);
    } else if(denoised.channels() == 4) {
        cv::cvtColor(denoised, denoisedDisplay, cv::COLOR_BGRA2GRAY);
    }
    
    // 10. Asegurar mismo tamaño
    if(denoisedDisplay.size() != currentImage.size()) {
        cv::resize(denoisedDisplay, denoisedDisplay, currentImage.size());
    }

    // 11. MOSTRAR EN LABEL_3 (Columna Red Neuronal)
    ui->label_4->setPixmap(QPixmap::fromImage(matToQImage(denoisedDisplay)
                          .scaled(ui->label_3->width(), ui->label_3->height(), 
                                 Qt::KeepAspectRatio, Qt::SmoothTransformation)));
    
    // 12. Guardar comparación en archivo
    cv::Mat origColor, denoisedColor;
    
    if(currentImage.channels() == 1) {
        cv::cvtColor(currentImage, origColor, cv::COLOR_GRAY2BGR);
    } else {
        origColor = currentImage.clone();
    }
    
    if(denoised.channels() == 1) {
        cv::cvtColor(denoised, denoisedColor, cv::COLOR_GRAY2BGR);
    } else if(denoised.channels() == 3) {
        denoisedColor = denoised.clone();
    } else if(denoised.channels() == 4) {
        cv::cvtColor(denoised, denoisedColor, cv::COLOR_BGRA2BGR);
    }
    
    if(origColor.size() != denoisedColor.size()) {
        cv::resize(denoisedColor, denoisedColor, origColor.size());
    }
    
    cv::Mat comparison;
    cv::hconcat(origColor, denoisedColor, comparison);
    
    cv::putText(comparison, "Original", cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    cv::putText(comparison, "DnCNN", 
                cv::Point(origColor.cols + 10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    
    QString comparisonPath = "output/dnn_comparison.png";
    cv::imwrite(comparisonPath.toStdString(), comparison);
    
    // 13. Limpiar temporal
    QFile::remove(tempImagePath);
    
    // 14. Mensaje de éxito
    QMessageBox::information(this, "DnCNN Denoising", 
                            "Denoising completado exitosamente!\n\n"
                            "Modelo: DnCNN Gray Blind (pre-entrenado)\n"
                            "Imagen mostrada en panel central\n"
                            "Comparacion guardada en: " + comparisonPath);
    dncnnImage = denoisedDisplay.clone();
    dncnnApplied = true;

    updateImageByCheckbox();
}

void MainWindow::updateImageByCheckbox()
{
    if (originalImage.empty()) {
        return;
    }

    if (ui->checkBox_dncnn_suavizado->isChecked()) {

        if (dncnnApplied == false) {
            on_pushButton_3_clicked();   
        }
        else {
            showImage(dncnnImage);
        }
    }
    else {
        dncnnApplied = false;
        showImage(originalImage);
    }
}


void MainWindow::showImage(const cv::Mat &img)
{
    if (img.empty()) return;

    QImage qimg = matToQImage(img);

    ui->label->setPixmap(
        QPixmap::fromImage(qimg).scaled(
            ui->label->width(),
            ui->label->height(),
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation
        )
    );
}


double MainWindow::getCurrentRAMUsageMB()
{
    std::ifstream statm("/proc/self/statm");
    long size = 0;
    long resident = 0;

    statm >> size >> resident;
    statm.close();

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024;

    double ramMB = (resident * page_size_kb) / 1024.0;
    return ramMB;
}


