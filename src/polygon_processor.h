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
    struct PolygonData {
        exchange_protocol::PolygonType type = exchange_protocol::PolygonType::INCLUDE;
        int priority = -1;
        std::vector<cv::Point> points;

        cv::Rect bbox;
        std::unordered_set<int> class_ids;
    };

    explicit PolygonProcessor(const std::vector<exchange_protocol::PolygonConfig>& polygons,
                              const std::unordered_map<std::string, int>& class_name_to_id)
    {
        for (auto poly : polygons) {
            if (poly.points().size() < 3) {
                throw std::invalid_argument(
                    "Polygon must have at least 3 points, got " + 
                    std::to_string(poly.points().size()));
            }
            PolygonData poly_data;

            for (const auto& class_name : poly.class_filters()) {
                auto it = class_name_to_id.find(class_name);
                if (it != class_name_to_id.end()) {
                    poly_data.class_ids.insert(it->second);
                }
            }

            poly_data.points.reserve(poly.points().size());
            for (const auto& vertex : poly.points()) {
                poly_data.points.emplace_back(vertex.x(), vertex.y());
            }

            poly_data.bbox = cv::boundingRect(poly_data.points);
            poly_data.type = poly.type();
            poly_data.priority = poly.priority();
            polygons_.push_back(std::move(poly_data));
        }
    }
    
    bool IsPointInPolygons(const cv::Point& point, int class_id) const {
        exchange_protocol::PolygonType highest_priority_type = exchange_protocol::PolygonType::EXCLUDE;
        int highest_priority = -1;

        for (const auto& poly : polygons_) {
            if (poly.class_ids.count(class_id) == 0) {
                continue;
            }
            if (!poly.bbox.contains(point)) {
                continue;  // Cheap test to avoid expensive pointPolygonTest
            }
            
            if (cv::pointPolygonTest(poly.points, point, false) >= 0) {
                if (poly.priority > highest_priority) {
                    highest_priority = poly.priority;
                    highest_priority_type = poly.type;
                } else if (poly.priority == highest_priority) {
                    if (poly.type == exchange_protocol::PolygonType::EXCLUDE) {
                        highest_priority_type = exchange_protocol::PolygonType::EXCLUDE;
                    }
                }
            }
        }

        if (highest_priority_type == exchange_protocol::PolygonType::INCLUDE) {
            return true;
        }
        return false;
    }

private:
    std::vector<PolygonData> polygons_;

};
#endif  // SRC_POLYGON_PROCESSOR_H_
