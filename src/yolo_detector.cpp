#include "yolo_detector.h"

#include <opencv2/dnn.hpp>

#include <fstream>
#include <iostream>

YoloDetector::YoloDetector(const std::string& cfg,
                           const std::string& weights,
                           const std::string& class_names_file,
                           float confidence_threshold,
                           float nms_threshold)
    : confidence_threshold_(confidence_threshold), nms_threshold_(nms_threshold)
{
    net_ = cv::dnn::readNetFromDarknet(cfg, weights);
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // Load class names
    std::ifstream ifs(class_names_file);
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty()) {
            class_names_.push_back(line);
        }
    }
}

std::vector<Detection> YoloDetector::Detect(const cv::Mat& frame)
{
    std::vector<Detection> detections;
    if (frame.empty()) {
        return detections;
    }

    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(416,416), cv::Scalar(), true, false);
    net_.setInput(blob);

    std::vector<cv::Mat> outs;
    net_.forward(outs, net_.getUnconnectedOutLayersNames());

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (const auto& out : outs) {
        for (int i = 0; i < out.rows; ++i) {
            const float* data = out.ptr<float>(i);
            float score = data[4];
            if (score < confidence_threshold_) {
                continue;
            }
            cv::Mat scores = out.row(i).colRange(5, out.cols);
            cv::Point classIdPoint;
            double confidence;
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confidence_threshold_) {
                int centerX = static_cast<int>(data[0] * frame.cols);
                int centerY = static_cast<int>(data[1] * frame.rows);
                int width   = static_cast<int>(data[2] * frame.cols);
                int height  = static_cast<int>(data[3] * frame.rows);
                int left    = centerX - width / 2;
                int top     = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back(static_cast<float>(confidence));
                boxes.emplace_back(left, top, width, height);
            }
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold_, nms_threshold_, indices);

    for (int idx : indices) {
        Detection det;
        det.class_id = classIds[idx];
        det.confidence = confidences[idx];
        det.bbox = boxes[idx];
        if (det.class_id >= 0 && det.class_id < static_cast<int>(class_names_.size())) {
            det.class_name = class_names_[det.class_id];
        } else {
            det.class_name = "unknown";
        }
        detections.push_back(det);
    }
    return detections;
}