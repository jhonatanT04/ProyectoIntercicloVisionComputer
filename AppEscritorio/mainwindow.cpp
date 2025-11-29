#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "pipelinedialog.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    processor = new CTImageProcessor("output");  
    procesado = new ImageProcessor();
    // Configurar rangos de sliders
    ui->horizontalSlider->setRange(100, 255);      
    ui->horizontalSlider->setValue(170);  //Parametro 1 Huesos 1-255 (inRange)

    ui->horizontalSlider_2->setRange(0, 30);    
    ui->horizontalSlider_2->setValue(0);  //Parametro 2 Huesos 0-30 (kernel Reduccion Ruido-Mediana)

    ui->horizontalSlider_3->setRange(0, 30);   
    ui->horizontalSlider_3->setValue(0); //Parametro 3 Huesos 0-30 (Kernel Suavizado-hightRegion)

    ui->horizontalSlider_4->setRange(0,255 );   
    ui->horizontalSlider_4->setValue(125);

    ui->horizontalSlider_5->setRange(1, 255);   
    ui->horizontalSlider_5->setValue(5);

    ui->horizontalSlider_6->setRange(1, 21);    
    ui->horizontalSlider_6->setValue(8);
    

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
    connect(ui->horizontalSlider_6, &QSlider::valueChanged,
            this, &MainWindow::updateFilters);
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
    // procesado->loadImage(fileName.toStdString());
    procesado->loadImage(fileName.toStdString());
    Mat img = procesado->getOriginalImage();
    // currentImage = processor->getOriginalImage();  
    currentImage = img.clone();  
    
    // Mostrar imagen original como icono en panel izquierdo
    QImage qorig = matToQImage(currentImage);
    ui->listWidget->clear();
    QListWidgetItem* item = new QListWidgetItem(
        QIcon(QPixmap::fromImage(qorig).scaled(100,100, Qt::KeepAspectRatio)),
        "Original"
    );
    ui->listWidget->addItem(item);

    // Aplicar pipeline interno automáticamente
    applyInternalPipeline();

    // Actualizar sliders en tiempo real
    updateFilters();
}

// ==================== Pipeline Interno OPTIMIZADO ====================
void MainWindow::applyInternalPipeline()
{
    if (currentImage.empty()) return;

    Mat img = currentImage.clone();
    Mat processed;

    // ============================================================
    // ETAPA 1: WINDOW/LEVEL PARA CT (¡CRÍTICO!)
    // ============================================================
    // Aplicar ventana de tejido blando si es imagen CT raw
    // processed = processor->applyWindowLevel(40, 400);
    processed = procesado->getOriginalImage();
    
    // ============================================================
    // ETAPA 2: MEJORA DE CONTRASTE (Stack completo)
    // ============================================================
    processed = procesado->normalizeImage(processed);
    processed = processor->applyCLAHE(processed, 3.0);
    processed = processor->histogramEqualization(processed);
    
    // ============================================================
    // ETAPA 3: FILTROS DE SUAVIZADO (orden estratégico)
    // ============================================================
    // 3.1 NL-Means: Denoise potente (primero porque es más robusto)
    processed = processor->filterNLMeans(processed);
    
    // 3.2 Bilateral: Preserva bordes mientras suaviza
    processed = processor->filterBilateral(processed, 5);
    
    // 3.3 Gaussian: Suavizado general
    processed = processor->filterGaussian(processed, 3);
    
    // 3.4 Median: Elimina ruido sal y pimienta (último para limpieza)
    processed = processor->filterMedian(processed, 3);
    
    // ============================================================
    // ETAPA 4: MORFOLOGÍA PARA LIMPIEZA
    // ============================================================
    // 4.1 Top-hat: Resalta estructuras brillantes pequeñas
    processed = processor->morphTopHat(processed, 5);
    
    // 4.2 Black-hat: Resalta estructuras oscuras pequeñas
    cv::Mat blackhat = processor->morphBlackHat(processed, 5);
    
    // 4.3 Combinar top-hat con la imagen procesada
    cv::Mat enhanced;
    cv::addWeighted(processed, 1.0, blackhat, 0.3, 0, enhanced);
    processed = enhanced;
    
    // 4.4 Opening: Elimina ruido pequeño
    processed = processor->morphOpening(processed, 3);
    
    // 4.5 Closing: Cierra huecos pequeños
    processed = processor->morphClosing(processed, 5);
    
    // ============================================================
    // ETAPA 5: SEGMENTACIÓN AUTOMÁTICA
    // ============================================================
    // 5.1 Threshold Otsu para segmentación automática
    cv::Mat segmented = processor->thresholdOtsu(processed);
    
    // 5.2 Threshold adaptativo (alternativa para comparar)
    cv::Mat adaptiveThresh = processor->thresholdAdaptive(processed, 11);
    
    // 5.3 Limpieza morfológica de la máscara
    segmented = processor->morphOpening(segmented, 3);
    segmented = processor->morphClosing(segmented, 7);
    
    // 5.4 Segmentación por intensidad (rango medio-alto)
    cv::Mat intensitySeg = processor->segmentByIntensity(processed, 100, 200);
    
    // 5.5 Combinar máscaras con OR
    cv::Mat finalMask;
    cv::bitwise_or(segmented, intensitySeg, finalMask);
    
    // ============================================================
    // ETAPA 6: DETECCIÓN DE BORDES
    // ============================================================
    cv::Mat edges = processor->edgeCanny(processed, 50, 150);
    
    // ============================================================
    // ETAPA 7: VISUALIZACIÓN AVANZADA
    // ============================================================
    // 7.1 Overlay verde sobre región segmentada
    cv::Mat overlay = processor->createColorOverlay(currentImage, finalMask, 
                                                     cv::Scalar(0, 255, 0), 0.4);
    
    // 7.2 Heatmap de intensidades
    cv::Mat heatmap = processor->createHeatmap(processed);
    
    // 7.3 Resaltar región con contornos
    cv::Mat highlighted = processor->highlightRegion(finalMask, currentImage);
    
    // 7.4 Overlay multicolor por rangos de intensidad
    cv::Mat multiOverlay;
    cv::cvtColor(currentImage, multiOverlay, cv::COLOR_GRAY2BGR);
    
    // Región baja intensidad (azul)
    cv::Mat lowIntensity = processor->segmentByIntensity(processed, 0, 100);
    cv::Mat blueOverlay = multiOverlay.clone();
    blueOverlay.setTo(cv::Scalar(255, 0, 0), lowIntensity);
    cv::addWeighted(multiOverlay, 0.8, blueOverlay, 0.2, 0, multiOverlay);
    
    // Región alta intensidad (rojo)
    cv::Mat highIntensity = processor->segmentByIntensity(processed, 200, 255);
    cv::Mat redOverlay = multiOverlay.clone();
    redOverlay.setTo(cv::Scalar(0, 0, 255), highIntensity);
    cv::addWeighted(multiOverlay, 0.8, redOverlay, 0.2, 0, multiOverlay);
    
    // ============================================================
    // ETAPA 8: IMAGEN FINAL COMBINADA
    // ============================================================
    cv::Mat finalCombined;
    cv::cvtColor(processed, finalCombined, cv::COLOR_GRAY2BGR);
    
    // Añadir bordes en verde
    cv::Mat edgeColor;
    cv::cvtColor(edges, edgeColor, cv::COLOR_GRAY2BGR);
    edgeColor.setTo(cv::Scalar(0, 255, 0), edges);
    cv::addWeighted(finalCombined, 0.7, edgeColor, 0.3, 0, finalCombined);
    
    // Guardar resultado del pipeline
    pipelineImage = overlay;
    
    // ============================================================
    // MOSTRAR RESULTADOS EN LA INTERFAZ
    // ============================================================
    // Label 1: Overlay con segmentación
    ui->label->setPixmap(QPixmap::fromImage(matToQImage(overlay)
                        .scaled(ui->label->width(), ui->label->height(), 
                               Qt::KeepAspectRatio, Qt::SmoothTransformation)));
    
    // Label 2: Máscara segmentada
    ui->label_2->setPixmap(QPixmap::fromImage(matToQImage(finalMask)
                          .scaled(ui->label_2->width(), ui->label_2->height(), 
                                 Qt::KeepAspectRatio, Qt::SmoothTransformation)));
    
    // Label 3: Heatmap o región resaltada
    ui->label_3->setPixmap(QPixmap::fromImage(matToQImage(heatmap)
                          .scaled(ui->label_3->width(), ui->label_3->height(), 
                                 Qt::KeepAspectRatio, Qt::SmoothTransformation)));
    
    // Agregar miniaturas al listWidget
    QListWidgetItem* item2 = new QListWidgetItem(
        QIcon(QPixmap::fromImage(matToQImage(processed)).scaled(100,100, Qt::KeepAspectRatio)),
        "Procesada"
    );
    ui->listWidget->addItem(item2);
    
    QListWidgetItem* item3 = new QListWidgetItem(
        QIcon(QPixmap::fromImage(matToQImage(finalMask)).scaled(100,100, Qt::KeepAspectRatio)),
        "Segmentada"
    );
    ui->listWidget->addItem(item3);
    
    QListWidgetItem* item4 = new QListWidgetItem(
        QIcon(QPixmap::fromImage(matToQImage(heatmap)).scaled(100,100, Qt::KeepAspectRatio)),
        "Heatmap"
    );
    ui->listWidget->addItem(item4);
}

// ==================== Update Filters (Tiempo Real) ====================
void MainWindow::updateFilters()
{
    if(currentImage.empty()) return;

    // ============================================================
    // OBTENER VALORES DE SLIDERS
    // ============================================================
    int gaussianK = std::max(1, ui->horizontalSlider->value() | 1);  // Forzar impar
    // int gaussianK = 3;  // Forzar impar
    int medianK   = std::max(1, ui->horizontalSlider_2->value() | 1); // Forzar impar
    double claheClip = ui->horizontalSlider_3->value() / 10.0;
    
    //================ Codigo deteccion Huesos ================
    int a_h = ui->horizontalSlider->value();      // 1-255
    int b_h = std::max(1, ui->horizontalSlider_2->value() | 1);  // Forzar impar
    int k_n = std::max(1, ui->horizontalSlider_3->value() | 1);  // Forzar impar



    // Sliders adicionales (si los agregaste en Qt Designer)
    // int threshValue = ui->horizontalSlider_4->value();  // 0-255
    // int morphSize = std::max(1, ui->horizontalSlider_5->value() | 1);  // Forzar impar
    
    // Valores por defecto si no tienes los sliders
    int threshValue = ui->horizontalSlider_4->value();
    int morphSize = std::max(1, ui->horizontalSlider_5->value() | 1);  // Forzar impar

    cv::Mat filtered = currentImage.clone();

    // ============================================================
    // PROCESAMIENTO LIGERO Y RÁPIDO
    // ============================================================
    Mat img = procesado->getOriginalImage();
    Mat imgCLAHE = processor->applyCLAHE(img, 3);
    Mat imgMejoramiento = processor->segmentByIntensity(imgCLAHE, a_h, 255);

    Mat imgMejSuavizada = processor->filterNLMeans(imgMejoramiento);
    Mat suavizada2 = processor->filterMedian(imgMejSuavizada,b_h);
    suavizada2 =  processor->morphDilation(suavizada2,k_n);

    //================ Codigo deteccion Huesos ================

    //namedWindow("Imagen Suavizada", WINDOW_AUTOSIZE);
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

    // 1. MEJORA DE CONTRASTE (ligera)
    filtered = processor->normalize(filtered);
    filtered = processor->applyCLAHE(filtered, claheClip);
    
    // 2. SUAVIZADO AJUSTABLE POR USUARIO
    filtered = processor->filterGaussian(filtered, gaussianK);
    filtered = processor->filterMedian(filtered, medianK);
    
    // 3. MORFOLOGÍA LIGERA (solo Top-hat y Black-hat)
    cv::Mat tophat = processor->morphTopHat(filtered, 3);
    cv::Mat blackhat = processor->morphBlackHat(filtered, 3);
    cv::addWeighted(filtered, 1.0, tophat, 0.5, 0, filtered);
    cv::addWeighted(filtered, 1.0, blackhat, 0.3, 0, filtered);
    
    // 4. THRESHOLD AJUSTABLE (simple)
    cv::Mat thresholded = processor->threshold(filtered, threshValue);
    
    // 5. LIMPIEZA MORFOLÓGICA DE MÁSCARA
    thresholded = processor->morphOpening(thresholded, morphSize);
    thresholded = processor->morphClosing(thresholded, morphSize);
    
    // 6. DETECCIÓN DE BORDES (Canny)
    cv::Mat edges = processor->edgeCanny(filtered, 50, 150);
    
    // 7. GRADIENTE MORFOLÓGICO (alternativa a Canny)
    cv::Mat morphGrad = processor->morphGradient(filtered, 3);
    
    // 8. VISUALIZACIÓN
    cv::Mat overlay = processor->createColorOverlay(currentImage, thresholded, 
                                                     cv::Scalar(0, 255, 0), 0.5);
    
    // 9. COMBINACIÓN FINAL CON BORDES
    cv::Mat finalVis;
    cv::cvtColor(filtered, finalVis, cv::COLOR_GRAY2BGR);
    cv::Mat edgeColor;
    cv::cvtColor(edges, edgeColor, cv::COLOR_GRAY2BGR);
    edgeColor.setTo(cv::Scalar(0, 255, 255), edges);  // Bordes amarillos
    cv::addWeighted(finalVis, 0.8, edgeColor, 0.2, 0, finalVis);


    Mat imgHuesos = procesado->deteccionHuesos(ui->horizontalSlider->value(),255);
    Mat imgPulmones = procesado->deteccionPulmones(ui->horizontalSlider->value(),255,3);
    Mat imgMusculos = procesado->deteccionMuscular(ui->horizontalSlider->value(),255);

    // ============================================================
    // MOSTRAR RESULTADOS
    // ============================================================
    ui->label->setPixmap(QPixmap::fromImage(matToQImage(procesado->filterMedian(img,3))
                        .scaled(ui->label->width(), ui->label->height(), 
                               Qt::KeepAspectRatio, Qt::SmoothTransformation)));

    ui->label_2->setPixmap(QPixmap::fromImage(matToQImage(procesado->createColorOverlay(img, imgMejoramiento, Scalar(0,0,255),0.8))
                          .scaled(ui->label_2->width(), ui->label_2->height(), 
                                 Qt::KeepAspectRatio, Qt::SmoothTransformation)));

    ui->label_3->setPixmap(QPixmap::fromImage(matToQImage(procesado->highlightRegion("Hueso",suavizada2, img,Scalar(255,0,255)))
                          .scaled(ui->label_3->width(), ui->label_3->height(), 
                                 Qt::KeepAspectRatio, Qt::SmoothTransformation)));
}

// ==================== BOTÓN 2: Aplicar Pipeline Completo ====================
void MainWindow::on_pushButton_2_clicked()
{
    if(currentImage.empty()) {
        QMessageBox::warning(this, "Error", "Primero cargue una imagen.");
        return;
    }

    cv::Mat img = currentImage.clone();
    std::vector<std::pair<QString, cv::Mat>> stages;

    cv::Mat temp1, temp2;

    // ============ STAGE 1: Preparación ============
    temp1 = processor->applyWindowLevel(40, 400); 
    stages.push_back({"1. Window Level (Soft Tissue)", temp1});
    
    // ============ STAGE 2: Mejora de Contraste ============
    temp1 = processor->normalize(temp1); 
    stages.push_back({"2. Normalize", temp1});
    
    temp2 = processor->applyCLAHE(temp1, 3.0); 
    stages.push_back({"3. CLAHE", temp2});
    
    temp1 = processor->histogramEqualization(temp2); 
    stages.push_back({"4. Histogram Equalization", temp1});
    
    temp2 = processor->contrastStretching(temp1);
    stages.push_back({"5. Contrast Stretching", temp2});

    // ============ STAGE 3: Operaciones Lógicas ============
    temp1 = processor->applyNOT(temp2); 
    stages.push_back({"6. NOT", temp1});
    
    cv::Mat mask1 = processor->threshold(temp2, 100);
    cv::Mat mask2 = processor->threshold(temp2, 150);
    
    temp1 = processor->applyAND(mask1, mask2); 
    stages.push_back({"7. AND (máscaras)", temp1});
    
    temp1 = processor->applyOR(mask1, mask2); 
    stages.push_back({"8. OR (máscaras)", temp1});
    
    temp1 = processor->applyXOR(mask1, mask2); 
    stages.push_back({"9. XOR (máscaras)", temp1});

    // ============ STAGE 4: Detección de Bordes ============
    temp1 = temp2.clone();
    temp2 = processor->edgeCanny(temp1, 50, 120); 
    stages.push_back({"10. Canny Edge", temp2});
    
    temp2 = processor->edgeSobel(temp1); 
    stages.push_back({"11. Sobel Edge", temp2});
    
    temp2 = processor->edgeLaplacian(temp1); 
    stages.push_back({"12. Laplacian Edge", temp2});

    // ============ STAGE 5: Filtros de Suavizado ============
    temp1 = processor->filterGaussian(temp1, 5); 
    stages.push_back({"13. Gaussian Filter", temp1});
    
    temp1 = processor->filterMedian(temp1, 5); 
    stages.push_back({"14. Median Filter", temp1});
    
    temp1 = processor->filterBilateral(temp1, 9); 
    stages.push_back({"15. Bilateral Filter", temp1});
    
    temp1 = processor->filterMean(temp1, 5); 
    stages.push_back({"16. Mean Filter", temp1});
    
    temp1 = processor->filterNLMeans(temp1); 
    stages.push_back({"17. NL-Means Denoising", temp1});

    // ============ STAGE 6: Morfología ============
    temp2 = processor->thresholdOtsu(temp1);
    stages.push_back({"18. Threshold Otsu", temp2});
    
    temp1 = processor->morphErosion(temp2, 5); 
    stages.push_back({"19. Erosion", temp1});
    
    temp1 = processor->morphDilation(temp2, 5); 
    stages.push_back({"20. Dilation", temp1});
    
    temp1 = processor->morphOpening(temp2, 5); 
    stages.push_back({"21. Opening", temp1});
    
    temp1 = processor->morphClosing(temp2, 5); 
    stages.push_back({"22. Closing", temp1});
    
    temp1 = processor->morphGradient(temp2, 5); 
    stages.push_back({"23. Morphological Gradient", temp1});
    
    temp1 = processor->morphTopHat(img, 15); 
    stages.push_back({"24. Top Hat", temp1});
    
    temp1 = processor->morphBlackHat(img, 15); 
    stages.push_back({"25. Black Hat", temp1});

    // ============ STAGE 7: Segmentación ============
    temp1 = processor->segmentByIntensity(temp2, 100, 200); 
    stages.push_back({"26. Segment by Intensity", temp1});
    
    temp1 = processor->thresholdAdaptive(temp2, 11);
    stages.push_back({"27. Adaptive Threshold", temp1});

    // ============ STAGE 8: Visualización ============
    temp2 = processor->createColorOverlay(img, temp1, cv::Scalar(0,255,0), 0.5); 
    stages.push_back({"28. Green Overlay", temp2});
    
    temp2 = processor->createHeatmap(temp1); 
    stages.push_back({"29. Heatmap", temp2});
    
    temp2 = processor->highlightRegion(temp1, img);
    stages.push_back({"30. Highlighted Contours", temp2});

    // ============ STAGE 9: Resultado Final ============
    pipelineImage = temp2;
    stages.push_back({"31. FINAL RESULT", pipelineImage});

    // Mostrar diálogo con todas las etapas
    PipelineDialog dlg(stages, this);
    dlg.exec();
}

// ==================== BOTÓN 3: Red Neuronal (Denoising DNN) ====================
void MainWindow::on_pushButton_3_clicked()
{
    if(currentImage.empty()) {
        QMessageBox::warning(this, "Error", "Cargue una imagen primero.");
        return;
    }

    // Aplicar denoising con DNN
    cv::Mat denoised = processor->applyDenoisingDNN(currentImage);

    // Comparación lado a lado
    cv::Mat comparison;
    cv::Mat origColor, denoisedColor;
    
    if(currentImage.channels() == 1) {
        cv::cvtColor(currentImage, origColor, cv::COLOR_GRAY2BGR);
    } else {
        origColor = currentImage.clone();
    }
    
    if(denoised.channels() == 1) {
        cv::cvtColor(denoised, denoisedColor, cv::COLOR_GRAY2BGR);
    } else {
        denoisedColor = denoised.clone();
    }
    
    cv::hconcat(origColor, denoisedColor, comparison);
    
    // Agregar texto
    cv::putText(comparison, "Original", cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    cv::putText(comparison, "DNN Denoised", cv::Point(origColor.cols + 10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

    // Mostrar en label_3
    ui->label_3->setPixmap(QPixmap::fromImage(matToQImage(comparison)
                          .scaled(ui->label_3->width(), ui->label_3->height(), 
                                 Qt::KeepAspectRatio, Qt::SmoothTransformation)));
    
    // También guardar como archivo
    cv::imwrite("output/dnn_comparison.png", comparison);
    
    QMessageBox::information(this, "DNN Denoising", 
                            "Denoising completado.\nComparación guardada en: output/dnn_comparison.png");
}