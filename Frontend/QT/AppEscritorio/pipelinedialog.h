#ifndef PIPLINEDIALOG_H
#define PIPLINEDIALOG_H

#include <QDialog>
#include <QLabel>
#include <QVBoxLayout>
#include <QScrollArea>
#include <QPushButton>
#include <opencv2/opencv.hpp>
#include <vector>

namespace Ui { class PipelineDialog; }

class PipelineDialog : public QDialog
{
    Q_OBJECT
public:
    explicit PipelineDialog(const std::vector<std::pair<QString, cv::Mat>>& stages, QWidget *parent = nullptr);

private slots:
    void saveFinalImage();

private:
    Ui::PipelineDialog *ui;
    QVBoxLayout* scrollLayout;
    cv::Mat finalImage;
};

#endif // PIPLINEDIALOG_H
