#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "pipelinedialog.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    processor = new CTImageProcessor("output");  

    // Conectar sliders para tiempo real
    connect(ui->horizontalSlider, &QSlider::valueChanged,
            this, &MainWindow::updateFilters);
    connect(ui->horizontalSlider_2, &QSlider::valueChanged,
            this, &MainWindow::updateFilters);
}

MainWindow::~MainWindow()
{
    delete processor;
    delete ui;
}

// ==================== Conversión Mat -> QImage ====================
QImage MainWindow::matToQImage(const cv::Mat &mat)
{
    if(mat.type() == CV_8UC1)
        return QImage(mat.data, mat.cols, mat.rows, mat.step,
                      QImage::Format_Grayscale8).copy();
    if(mat.type() == CV_8UC3) {
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
        return QImage(rgb.data, rgb.cols, rgb.rows, rgb.step,
                      QImage::Format_RGB888).copy();
    }
    return QImage();
}

// ==================== BOTÓN 1: Cargar Imagen ====================
void MainWindow::on_pushButton_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(
        this, "Seleccionar imagen CT", "",
        "Imagenes (*.png *.jpg *.jpeg *.bmp *.IMA *.dcm)");

    if(fileName.isEmpty()) return;

    if (!processor->loadImage(fileName.toStdString())) {
        QMessageBox::warning(this, "Error", "No se pudo cargar la imagen.");
        return;
    }

    currentImage = processor->getOriginalImage();  

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

// ==================== Pipeline interno ====================
void MainWindow::applyInternalPipeline()
{
    cv::Mat img = currentImage.clone();
    cv::Mat temp1, temp2;

    // -------------------- Filtros esenciales --------------------
    temp1 = processor->normalize(img);
    temp2 = processor->applyCLAHE(temp1, 3.0);
    temp1 = processor->histogramEqualization(temp2);

    temp1 = processor->filterGaussian(temp1, 3);
    temp1 = processor->filterMedian(temp1, 3);

    temp1 = processor->morphTopHat(temp1, 3);  
    temp1 = processor->morphBlackHat(temp1, 3);

    temp1 = processor->segmentByIntensity(temp1, 100, 200);

    temp2 = processor->createColorOverlay(img, temp1, cv::Scalar(0,255,0), 0.5);
    pipelineImage = temp2;

    // Mostrar en el label grande
    ui->label->setPixmap(QPixmap::fromImage(matToQImage(pipelineImage)
                                            .scaled(ui->label->width(),
                                                    ui->label->height(),
                                                    Qt::KeepAspectRatio)));

    // Mostrar segmentada
    ui->label_2->setPixmap(QPixmap::fromImage(matToQImage(temp1)
                                              .scaled(ui->label_2->width(),
                                                      ui->label_2->height(),
                                                      Qt::KeepAspectRatio)));

    // Mostrar resaltada
    ui->label_3->setPixmap(QPixmap::fromImage(matToQImage(processor->highlightRegion(temp1))
                                              .scaled(ui->label_3->width(),
                                                      ui->label_3->height(),
                                                      Qt::KeepAspectRatio)));
}

// ==================== Actualizar filtros de sliders ====================
void MainWindow::updateFilters()
{
    if(pipelineImage.empty()) return;

    int gaussianK = std::max(1, ui->horizontalSlider->value() | 1);
    int medianK   = std::max(1, ui->horizontalSlider_2->value() | 1);
    double claheClip = ui->horizontalSlider_3->value() / 10.0;

    cv::Mat filtered = currentImage.clone();
    filtered = processor->normalize(filtered);
    filtered = processor->applyCLAHE(filtered, claheClip);
    filtered = processor->filterGaussian(filtered, gaussianK);
    filtered = processor->filterMedian(filtered, medianK);

    filtered = processor->morphTopHat(filtered, 3);
    filtered = processor->morphBlackHat(filtered, 3);

    cv::Mat segmented = processor->segmentByIntensity(filtered, 100, 200);
    cv::Mat highlighted = processor->createColorOverlay(currentImage, segmented, cv::Scalar(0,255,0), 0.5);

    ui->label->setPixmap(QPixmap::fromImage(matToQImage(highlighted)
                                            .scaled(ui->label->width(),
                                                    ui->label->height(),
                                                    Qt::KeepAspectRatio)));

    ui->label_2->setPixmap(QPixmap::fromImage(matToQImage(segmented)
                                              .scaled(ui->label_2->width(),
                                                      ui->label_2->height(),
                                                      Qt::KeepAspectRatio)));

    ui->label_3->setPixmap(QPixmap::fromImage(matToQImage(filtered)
                                              .scaled(ui->label_3->width(),
                                                      ui->label_3->height(),
                                                      Qt::KeepAspectRatio)));
}

// ==================== BOTÓN 2: Aplicar Pipeline completo ====================
void MainWindow::on_pushButton_2_clicked()
{
    if(currentImage.empty()) {
        QMessageBox::warning(this, "Error", "Primero cargue una imagen.");
        return;
    }

    cv::Mat img = currentImage.clone();
    std::vector<std::pair<QString, cv::Mat>> stages;

    cv::Mat temp1, temp2;

    temp1 = processor->normalize(img); stages.push_back({"Normalize", temp1});
    temp2 = processor->applyCLAHE(temp1, 3.0); stages.push_back({"CLAHE", temp2});
    temp1 = processor->histogramEqualization(temp2); stages.push_back({"Histogram Equalization", temp1});

    temp2 = processor->applyNOT(temp1); stages.push_back({"NOT", temp2});
    temp1 = processor->applyAND(temp1, temp2); stages.push_back({"AND", temp1});

    temp2 = processor->edgeCanny(temp1, 50, 120); stages.push_back({"Canny", temp2});
    temp1 = processor->edgeSobel(temp1); stages.push_back({"Sobel", temp1});
    temp1 = processor->edgeLaplacian(temp1); stages.push_back({"Laplacian", temp1});

    temp1 = processor->filterGaussian(temp1, 3); stages.push_back({"Gaussian", temp1});
    temp1 = processor->filterMedian(temp1, 3); stages.push_back({"Median", temp1});
    temp1 = processor->filterBilateral(temp1, 5); stages.push_back({"Bilateral", temp1});
    temp1 = processor->filterMean(temp1, 3); stages.push_back({"Mean", temp1});
    temp1 = processor->filterNLMeans(temp1); stages.push_back({"NLMeans", temp1});

    temp1 = processor->morphErosion(temp1, 3); stages.push_back({"Erosion", temp1});
    temp1 = processor->morphDilation(temp1, 3); stages.push_back({"Dilation", temp1});
    temp1 = processor->morphOpening(temp1, 3); stages.push_back({"Opening", temp1});
    temp1 = processor->morphClosing(temp1, 3); stages.push_back({"Closing", temp1});
    temp1 = processor->morphGradient(temp1, 3); stages.push_back({"Gradient", temp1});
    temp1 = processor->morphTopHat(temp1, 3); stages.push_back({"TopHat", temp1});
    temp1 = processor->morphBlackHat(temp1, 3); stages.push_back({"BlackHat", temp1});

    temp1 = processor->segmentByIntensity(temp1, 100, 200); stages.push_back({"Segmented", temp1});

    temp2 = processor->createColorOverlay(img, temp1, cv::Scalar(0,255,0), 0.5); stages.push_back({"Overlay", temp2});
    temp1 = processor->createHeatmap(temp1); stages.push_back({"Heatmap", temp1});

    pipelineImage = temp2;
    stages.push_back({"Final Pipeline", pipelineImage});

    PipelineDialog dlg(stages, this);
    dlg.exec();
}

// ==================== BOTÓN 3: Red Neuronal ====================
void MainWindow::on_pushButton_3_clicked()
{
    if(currentImage.empty()) {
        QMessageBox::warning(this, "Error", "Cargue una imagen primero.");
        return;
    }

    cv::Mat denoised = processor->applyDenoisingDNN(currentImage);

    ui->label_3->setPixmap(QPixmap::fromImage(matToQImage(denoised)
                                              .scaled(ui->label_3->width(),
                                                      ui->label_3->height(),
                                                      Qt::KeepAspectRatio)));
}
