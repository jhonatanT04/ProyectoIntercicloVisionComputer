#include "pipelinedialog.h"
#include "ui_pipelinedialog.h"
#include <QPixmap>
#include <QFileDialog>
#include <QMessageBox>

PipelineDialog::PipelineDialog(const std::vector<std::pair<QString, cv::Mat>>& stages, QWidget *parent)
    : QDialog(parent), ui(new Ui::PipelineDialog)
{
    ui->setupUi(this);
    QGridLayout* gridLayout = findChild<QGridLayout*>("gridLayout");

    int row = 0, col = 0;
    const int maxCols = 3; // máximo de columnas por fila

    for(const auto& stage : stages) {
        // Texto del filtro
        QLabel* label = new QLabel(stage.first, this);
        label->setAlignment(Qt::AlignCenter);
        label->setStyleSheet("font-weight:bold; background-color:#ECEFF4; margin:5px; padding:5px;");
        gridLayout->addWidget(label, row, col);

        // Imagen
        QLabel* imageLabel = new QLabel(this);
        QImage qimg;
        if(stage.second.type() == CV_8UC1)
            qimg = QImage(stage.second.data, stage.second.cols, stage.second.rows, stage.second.step, QImage::Format_Grayscale8).copy();
        else if(stage.second.type() == CV_8UC3) {
            cv::Mat rgb;
            cv::cvtColor(stage.second, rgb, cv::COLOR_BGR2RGB);
            qimg = QImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888).copy();
        }
        imageLabel->setPixmap(QPixmap::fromImage(qimg).scaled(200, 200, Qt::KeepAspectRatio));
        imageLabel->setAlignment(Qt::AlignCenter);
        gridLayout->addWidget(imageLabel, row+1, col);

        col++;
        if(col >= maxCols) { col = 0; row += 2; } // subir fila cada 2 filas (texto + imagen)

        finalImage = stage.second; // última imagen
    }

    connect(findChild<QPushButton*>("saveButton"), &QPushButton::clicked, this, &PipelineDialog::saveFinalImage);
    connect(findChild<QPushButton*>("closeButton"), &QPushButton::clicked, this, &PipelineDialog::close);
}

void PipelineDialog::saveFinalImage()
{
    QString fileName = QFileDialog::getSaveFileName(this, "Guardar Imagen Final", "", "PNG Files (*.png)");
    if(fileName.isEmpty()) return;
    cv::imwrite(fileName.toStdString(), finalImage);
    QMessageBox::information(this, "Guardado", "Imagen final guardada correctamente.");
}
