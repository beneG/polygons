#ifndef SRC_YOLO_DETECTOR_H_
#define SRC_YOLO_DETECTOR_H_

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

//namespace object_detection {

/**
 * @brief Represents a detected object with bounding box and class information
 */
struct Detection {
  int class_id;           ///< Class ID from COCO dataset
  std::string class_name; ///< Human-readable class name
  float confidence;       ///< Detection confidence score (0.0 - 1.0)
  cv::Rect bbox;          ///< Bounding box coordinates
  cv::Point center;       ///< Center point of the bounding box
};

/**
 * @brief YOLO-based object detector using OpenCV DNN module
 * 
 * This class loads YOLOv3 model and performs object detection
 * on input images. It supports filtering by confidence threshold
 * and Non-Maximum Suppression (NMS).
 */
class YoloDetector {
 public:
  /**
   * @brief Constructs a YoloDetector with model files
   * 
   * @param model_config Path to YOLO configuration file (.cfg)
   * @param model_weights Path to YOLO weights file (.weights)
   * @param class_names_file Path to file with class names
   * @param confidence_threshold Minimum confidence for detections (default: 0.5)
   * @param nms_threshold Non-Maximum Suppression threshold (default: 0.4)
   */
  YoloDetector(const std::string& model_config,
               const std::string& model_weights,
               const std::string& class_names_file,
               float confidence_threshold = 0.5f,
               float nms_threshold = 0.4f);

  /**
   * @brief Detects objects in the given image
   * 
   * @param image Input image in BGR format
   * @return Vector of detected objects
   */
  std::vector<Detection> Detect(const cv::Mat& image);

  /**
   * @brief Gets the list of class names
   * 
   * @return Vector of class names
   */
  const std::vector<std::string>& GetClassNames() const {
    return class_names_;
  }

  /**
   * @brief Sets confidence threshold
   * 
   * @param threshold New confidence threshold value (0.0 - 1.0)
   */
  void SetConfidenceThreshold(float threshold) {
    confidence_threshold_ = threshold;
  }

  /**
   * @brief Sets NMS threshold
   * 
   * @param threshold New NMS threshold value (0.0 - 1.0)
   */
  void SetNmsThreshold(float threshold) {
    nms_threshold_ = threshold;
  }

 private:
  /**
   * @brief Loads class names from file
   * 
   * @param filename Path to file with class names (one per line)
   */
  void LoadClassNames(const std::string& filename);

  /**
   * @brief Gets output layer names for YOLO model
   * 
   * @return Vector of output layer names
   */
  std::vector<std::string> GetOutputLayerNames();

  cv::dnn::Net net_;                      ///< OpenCV DNN network
  std::vector<std::string> class_names_;  ///< List of class names
  float confidence_threshold_;            ///< Confidence threshold
  float nms_threshold_;                   ///< NMS threshold
  
  static constexpr int kInputWidth = 416;   ///< YOLO input width
  static constexpr int kInputHeight = 416;  ///< YOLO input height
};

//}  // namespace object_detection

#endif  // SRC_YOLO_DETECTOR_H_