#ifndef SRC_IDETECTOR_H_
#define SRC_IDETECTOR_H_

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include "proto/exchange_protocol.pb.h"


/**
 * @brief Represents a detected object with bounding box and class information
 */
struct Detection {
  int class_id;            ///< Class ID from COCO dataset
  std::string class_name;  ///< Human-readable class name
  float confidence;        ///< Detection confidence score (0.0 - 1.0)
  cv::Rect bbox;           ///< Bounding box coordinates
  cv::Point center;        ///< Center point of the bounding box
};


class IObjectDetector {
public:
    virtual ~IObjectDetector() = default;
    virtual std::vector<Detection> Detect(
        const cv::Mat& image,
        const std::vector<exchange_protocol::PolygonConfig>& polygons) = 0;
};

#endif  // SRC_IDETECTOR_H_