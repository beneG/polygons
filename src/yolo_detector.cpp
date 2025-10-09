#include "yolo_detector.h"
#include "polygon_processor.h"

#include <opencv2/dnn.hpp>

#include <fstream>
#include <iostream>

YoloDetector::YoloDetector(const std::string& model_config_file,
                           const std::string& model_weights_file,
                           const std::string& class_names_file,
                           float confidence_threshold,
                           float nms_threshold)
    : confidence_threshold_(confidence_threshold), nms_threshold_(nms_threshold)
{
    net_ = cv::dnn::readNetFromDarknet(model_config_file, model_weights_file);
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    LoadClassNames(class_names_file);
}

void YoloDetector::LoadClassNames(const std::string& class_names_file)
{
    std::ifstream ifs(class_names_file);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open class names file: " + class_names_file);
    }
    std::string line;
    int id = 0;
    while (std::getline(ifs, line)) {
        if (!line.empty()) {
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);
            class_name_to_id_[line] = id++;
            class_names_.push_back(std::move(line));
        }
    }
}


std::vector<Detection> YoloDetector::Detect(const cv::Mat& frame, const std::vector<exchange_protocol::PolygonConfig>& polygons)
{
    std::vector<Detection> detections;
    if (frame.empty()) {
        return detections;
    }

    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(kInputWidth, kInputHeight), cv::Scalar(), true, false);
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

    PolygonProcessor processor(polygons, class_name_to_id_);

    for (int idx : indices) {
        Detection det;
        det.class_id = classIds[idx];
        det.confidence = confidences[idx];
        det.bbox = boxes[idx];

        det.center = cv::Point(det.bbox.x + det.bbox.width / 2, det.bbox.y + det.bbox.height / 2);

        if (!processor.IsPointInPolygons(det.center, det.class_id)) {
            continue;
        }

        if (det.class_id >= 0 && det.class_id < static_cast<int>(class_names_.size())) {
            det.class_name = class_names_[det.class_id];
        } else {
            det.class_name = "unknown";
        }
        detections.push_back(std::move(det));
    }
    return detections;
}