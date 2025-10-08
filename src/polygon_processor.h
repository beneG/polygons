#ifndef SRC_POLYGON_PROCESSOR_H_
#define SRC_POLYGON_PROCESSOR_H_

#include <vector>
#include <opencv2/opencv.hpp>
#include "proto/exchange_protocol.pb.h"
#include "yolo_detector.h"
/**
 * @brief Processes polygons and filters detections
 * 
 * This class handles polygon definitions, checks if detections
 * fall within any of the polygons, and draws polygons on images.
 */
class PolygonProcessor {
    public:
    explicit PolygonProcessor(const std::vector<exchange_protocol::PolygonConfig>& polygons)
        : polygons_(polygons) {}
    
    // Filters detections to only those within any of the defined polygons
    std::vector<Detection> FilterDetections(const std::vector<Detection>& detections) {
        std::vector<Detection> filtered;
        for (const auto& det : detections) {
        cv::Point center(det.bbox.x + det.bbox.width / 2, 
                        det.bbox.y + det.bbox.height / 2);
        if (IsPointInAnyPolygon(center)) {
            filtered.push_back(det);
        }
        }
        return filtered;
    }
    
    // Draws the defined polygons on the image
    void DrawPolygons(cv::Mat& image) {
        for (const auto& poly : polygons_) {
        std::vector<cv::Point> points;
        for (const auto& vertex : poly.points()) {
            points.emplace_back(vertex.x(), vertex.y());
        }
        const cv::Point* pts = points.data();
        int npts = static_cast<int>(points.size());
        cv::polylines(image, &pts, &npts, 1, true, cv::Scalar(0, 255, 0), 2);
        }
    }
    
    private:
    std::vector<exchange_protocol::PolygonConfig> polygons_;
    
    // Checks if a point is inside any of the defined polygons
    bool IsPointInAnyPolygon(const cv::Point& point) {
        for (const auto& poly : polygons_) {
        std::vector<cv::Point> points;
        for (const auto& vertex : poly.points()) {
            points.emplace_back(vertex.x(), vertex.y());
        }
        if (cv::pointPolygonTest(points, point, false) >= 0) {
            return true;
        }
        }
        return false;
    }
};
#endif  // SRC_POLYGON_PROCESSOR_H_
