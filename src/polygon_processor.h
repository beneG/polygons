#ifndef SRC_POLYGON_PROCESSOR_H_
#define SRC_POLYGON_PROCESSOR_H_

#include <vector>
#include <opencv2/opencv.hpp>
#include "proto/exchange_protocol.pb.h"
/**
 * @brief Processes polygons and filters detections
 * 
 * This class handles polygon definitions, checks if detections
 * fall within any of the polygons, and draws polygons on images.
 */
class PolygonProcessor {
public:
    explicit PolygonProcessor(const std::vector<exchange_protocol::PolygonConfig>& polygons,
                              const std::unordered_map<std::string, int>& class_name_to_id)
        : polygons_(polygons)
    {
        for (auto poly : polygons_) {
            std::unordered_set<int> class_ids;
            for (const auto& class_name : poly.class_filters()) {
                auto it = class_name_to_id.find(class_name);
                if (it != class_name_to_id.end()) {
                    class_ids.insert(it->second);
                }
            }
            polygons_class_ids_.push_back(std::move(class_ids));

            std::vector<cv::Point> points(poly.points().size());
            for (const auto& vertex : poly.points()) {
                points.emplace_back(vertex.x(), vertex.y());
            }
            polygons_points_.push_back(std::move(points));
        }
    }
    
    /*
    
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
    */

    bool IsPointInPolygons(const cv::Point& point, int class_id) {

        exchange_protocol::PolygonType highest_priority_type = exchange_protocol::PolygonType::EXCLUDE;
        int highest_priority = -1;
        int poly_index = 0;

        for (const auto& poly : polygons_) {
            if (polygons_class_ids_[poly_index].count(class_id) > 0
                && cv::pointPolygonTest(polygons_points_[poly_index], point, false) >= 0) {

                if (poly.priority() > highest_priority) {
                    highest_priority = poly.priority();
                    highest_priority_type = poly.type();
                } else if (poly.priority() == highest_priority) {
                    // If same priority, EXCLUDE takes precedence over INCLUDE
                    if (poly.type() == exchange_protocol::PolygonType::EXCLUDE) {
                        highest_priority_type = exchange_protocol::PolygonType::EXCLUDE;
                    }
                }
            }
            poly_index++;
        }

        if (highest_priority_type == exchange_protocol::PolygonType::INCLUDE) {
            return true;
        }
        return false;
    }

private:
    std::vector<exchange_protocol::PolygonConfig> polygons_;
    std::vector<std::unordered_set<int>> polygons_class_ids_;
    std::vector<std::vector<cv::Point>> polygons_points_;

};
#endif  // SRC_POLYGON_PROCESSOR_H_
