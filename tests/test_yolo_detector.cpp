#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>

#include "proto/exchange_protocol.pb.h"
#include "yolo_detector.h"

namespace {

class YoloDetectorTest : public ::testing::Test {
 protected:
  std::string model_config_;
  std::string model_weights_;
  std::string class_names_;

  void SetUp() override {
    model_config_ = "data/models/yolov3.cfg";
    model_weights_ = "data/models/yolov3.weights";
    class_names_ = "data/models/coco.names";
  }

  // Helper to create test image
  cv::Mat CreateTestImage(int width = 640, int height = 480) {
    return cv::Mat(height, width, CV_8UC3, cv::Scalar(128, 128, 128));
  }

  // Helper to create polygon config
  exchange_protocol::PolygonConfig CreatePolygonConfig(
      const std::vector<std::pair<int, int>>& points,
      exchange_protocol::PolygonType type, int priority,
      const std::vector<std::string>& class_filters) {
    exchange_protocol::PolygonConfig config;
    for (const auto& p : points) {
      auto* point = config.add_points();
      point->set_x(p.first);
      point->set_y(p.second);
    }
    config.set_type(type);
    config.set_priority(priority);
    for (const auto& filter : class_filters) {
      config.add_class_filters(filter);
    }
    return config;
  }
};

// Test 1: Constructor loads model successfully
TEST_F(YoloDetectorTest, ConstructorLoadsModel) {
  EXPECT_NO_THROW(
      { YoloDetector detector(model_config_, model_weights_, class_names_); });
}

// Test 2: Constructor throws on invalid config file
TEST_F(YoloDetectorTest, ConstructorThrowsOnInvalidConfig) {
  EXPECT_THROW(
      { YoloDetector detector("invalid.cfg", model_weights_, class_names_); },
      cv::Exception);
}

// Test 3: Constructor throws on invalid weights file
TEST_F(YoloDetectorTest, ConstructorThrowsOnInvalidWeights) {
  EXPECT_THROW(
      {
        YoloDetector detector(model_config_, "invalid.weights", class_names_);
      },
      cv::Exception);
}

// Test 4: Constructor throws on invalid class names file
TEST_F(YoloDetectorTest, ConstructorThrowsOnInvalidClassNames) {
  EXPECT_THROW(
      {
        YoloDetector detector(model_config_, model_weights_, "invalid.names");
      },
      std::runtime_error);
}

// Test 5: Get class names returns non-empty list
TEST_F(YoloDetectorTest, GetClassNamesReturnsNonEmpty) {
  YoloDetector detector(model_config_, model_weights_, class_names_);
  const auto& class_names = detector.GetClassNames();

  EXPECT_FALSE(class_names.empty());
  EXPECT_GT(class_names.size(), 0);
  // COCO dataset has 80 classes
  EXPECT_EQ(class_names.size(), 80);
}

// Test 6: Detect on empty image returns empty detections
TEST_F(YoloDetectorTest, DetectOnEmptyImageReturnsEmpty) {
  YoloDetector detector(model_config_, model_weights_, class_names_);
  cv::Mat empty_image;
  std::vector<exchange_protocol::PolygonConfig> polygons;

  auto detections = detector.Detect(empty_image, polygons);
  EXPECT_TRUE(detections.empty());
}

// Test 7: Detect with no polygons processes all detections
TEST_F(YoloDetectorTest, DetectWithNoPolygons) {
  YoloDetector detector(model_config_, model_weights_, class_names_, 0.5f,
                        0.4f);
  cv::Mat test_image = CreateTestImage();
  std::vector<exchange_protocol::PolygonConfig> empty_polygons;

  // Should not crash, but may return 0 detections on uniform gray image
  EXPECT_NO_THROW({
    auto detections = detector.Detect(test_image, empty_polygons);
    // Gray image unlikely to have detections, but shouldn't crash
  });
}

// Test 8: Set confidence threshold
TEST_F(YoloDetectorTest, SetConfidenceThreshold) {
  YoloDetector detector(model_config_, model_weights_, class_names_);

  EXPECT_NO_THROW({
    detector.SetConfidenceThreshold(0.3f);
    detector.SetConfidenceThreshold(0.7f);
  });
}

// Test 9: Set NMS threshold
TEST_F(YoloDetectorTest, SetNmsThreshold) {
  YoloDetector detector(model_config_, model_weights_, class_names_);

  EXPECT_NO_THROW({
    detector.SetNmsThreshold(0.3f);
    detector.SetNmsThreshold(0.5f);
  });
}

// Test 10: Detect with INCLUDE polygon
TEST_F(YoloDetectorTest, DetectWithIncludePolygon) {
  YoloDetector detector(model_config_, model_weights_, class_names_, 0.5f,
                        0.4f);
  cv::Mat test_image = CreateTestImage();

  std::vector<exchange_protocol::PolygonConfig> polygons;
  polygons.push_back(CreatePolygonConfig(
      {{100, 100}, {500, 100}, {500, 400}, {100, 400}},
      exchange_protocol::PolygonType::INCLUDE, 1, {"person", "car"}));

  EXPECT_NO_THROW({
    auto detections = detector.Detect(test_image, polygons);
    // Detections should be filtered by polygon
  });
}

// Test 11: Detection structure has valid fields
TEST_F(YoloDetectorTest, DetectionStructureValidation) {
  // This test validates the Detection structure is properly populated
  // We can't guarantee detections on a gray image, but we can check
  // that if detections exist, they have valid fields

  YoloDetector detector(model_config_, model_weights_, class_names_, 0.1f,
                        0.4f);

  // Create a more complex image that might trigger detections
  cv::Mat test_image = CreateTestImage(640, 480);
  cv::rectangle(test_image, cv::Rect(100, 100, 200, 300), cv::Scalar(255, 0, 0),
                cv::FILLED);

  std::vector<exchange_protocol::PolygonConfig> polygons;
  polygons.push_back(
      CreatePolygonConfig({{0, 0}, {640, 0}, {640, 480}, {0, 480}},
                          exchange_protocol::PolygonType::INCLUDE, 1,
                          {"person", "car", "dog", "cat", "bicycle"}));

  auto detections = detector.Detect(test_image, polygons);

  for (const auto& det : detections) {
    // Validate detection fields
    EXPECT_GE(det.class_id, 0);
    EXPECT_LT(det.class_id, 80);  // COCO has 80 classes
    EXPECT_FALSE(det.class_name.empty());
    EXPECT_GE(det.confidence, 0.0f);
    EXPECT_LE(det.confidence, 1.0f);
    EXPECT_GE(det.bbox.width, 0);
    EXPECT_GE(det.bbox.height, 0);
    // Center should be within bbox
    EXPECT_GE(det.center.x, det.bbox.x);
    EXPECT_LE(det.center.x, det.bbox.x + det.bbox.width);
    EXPECT_GE(det.center.y, det.bbox.y);
    EXPECT_LE(det.center.y, det.bbox.y + det.bbox.height);
  }
}

// Test 12: Different confidence thresholds affect detection count
TEST_F(YoloDetectorTest, ConfidenceThresholdAffectsDetections) {
  cv::Mat test_image = CreateTestImage(640, 480);
  cv::rectangle(test_image, cv::Rect(100, 100, 200, 300),
                cv::Scalar(255, 128, 0), cv::FILLED);

  std::vector<exchange_protocol::PolygonConfig> polygons;
  polygons.push_back(
      CreatePolygonConfig({{0, 0}, {640, 0}, {640, 480}, {0, 480}},
                          exchange_protocol::PolygonType::INCLUDE, 1,
                          {"person", "car", "dog", "cat", "bicycle", "truck",
                           "bus", "motorcycle"}));

  YoloDetector detector_low(model_config_, model_weights_, class_names_, 0.1f,
                            0.4f);
  YoloDetector detector_high(model_config_, model_weights_, class_names_, 0.9f,
                             0.4f);

  auto detections_low = detector_low.Detect(test_image, polygons);
  auto detections_high = detector_high.Detect(test_image, polygons);

  // Lower threshold should give more or equal detections
  EXPECT_GE(detections_low.size(), detections_high.size());
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}