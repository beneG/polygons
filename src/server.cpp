#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <thread>
#include <chrono>
#include <csignal>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>
#include <cxxopts.hpp>

#include "proto/exchange_protocol.grpc.pb.h"
#include "proto/exchange_protocol.pb.h"
#include "yolo_detector.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using grpc::StatusCode;

using exchange_protocol::DetectionRequest;
using exchange_protocol::DetectionResponse;
using exchange_protocol::ObjectDetectorService;

/**
 * @brief Implementation of ObjectDetectorService
 */
class ObjectDetectorServiceImpl final : public ObjectDetectorService::Service {
 public:
  /**
   * @brief Constructs service implementation with YOLO detector
   *
   * @param detector Shared pointer to YOLO detector instance
   */
  explicit ObjectDetectorServiceImpl(std::shared_ptr<IObjectDetector> detector)
      : detector_(std::move(detector)) {}

  /**
   * @brief Handles object detection RPC requests
   *
   * @param context gRPC server context
   * @param request Detection request with image and polygons
   * @param response Detection response with result image
   * @return gRPC status
   */
  Status DetectObjects(ServerContext* context, const DetectionRequest* request,
                       DetectionResponse* response) override {

    spdlog::info("Received DetectObjects request: image size={} bytes, polygons={}",
                 request->image_data().size(), request->polygons_size());

    try {
      // Decode image from request
      cv::Mat raw_data(1, request->image_data().size(), CV_8UC1,
                       (void*)request->image_data().data());
      cv::Mat image = cv::imdecode(raw_data, cv::IMREAD_COLOR);

      if (image.empty()) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      "Failed to decode input image");
      }

      spdlog::info("Image decoded: {}x{}", image.cols, image.rows);

      std::vector<exchange_protocol::PolygonConfig> polygons(
          request->polygons().begin(), request->polygons().end());

      // Perform object detection
      std::vector<Detection> detections = detector_->Detect(image, polygons);
      spdlog::info("Detected {} objects", detections.size());

      DrawDetections(image, detections);
      DrawPolygons(image, polygons);

      // Encode result image
      std::vector<uchar> encoded_image;
      std::vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY, 95};

      if (!cv::imencode(".jpg", image, encoded_image, encode_params)) {
        return Status(StatusCode::INTERNAL, "Failed to encode result image");
      }

      response->set_result_image_data(encoded_image.data(),
                                      encoded_image.size());

      spdlog::info("Detection completed successfully. Returning {} objects",
                   detections.size());

      return Status::OK;

    } catch (const std::exception& e) {
      spdlog::error("Error during detection: {}", e.what());
      return Status(StatusCode::INTERNAL,
                    std::string("Detection error: ") + e.what());
    }
  }

 private:
  /**
   * @brief Draws detection bounding boxes and labels on image
   *
   * @param image Image to draw on
   * @param detections Vector of detections to draw
   */
  void DrawDetections(cv::Mat& image,
                      const std::vector<Detection>& detections) const {
    for (const auto& det : detections) {
      // Draw bounding box
      cv::rectangle(image, det.bbox, cv::Scalar(255, 178, 50), 2);

      // Prepare label
      std::string label =
          det.class_name + " " +
          std::to_string(static_cast<int>(det.confidence * 100)) + "%";

      // Draw label background
      int baseline;
      cv::Size label_size =
          cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

      int label_y = std::max(det.bbox.y, label_size.height);
      cv::rectangle(
          image, cv::Point(det.bbox.x, label_y - label_size.height),
          cv::Point(det.bbox.x + label_size.width, label_y + baseline),
          cv::Scalar(255, 178, 50), cv::FILLED);

      // Draw label text
      cv::putText(image, label, cv::Point(det.bbox.x, label_y),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
  }

  /**
   * @brief Draws polygons on the image (EXCLUDE=red, INCLUDE=green)
   *
   * @param image Image to draw on
   * @param polygons Vector of polygons to draw
   */
  void DrawPolygons(
      cv::Mat& image,
      const std::vector<exchange_protocol::PolygonConfig>& polygons) const {

    std::vector<cv::Point> points;
    points.reserve(20);

    for (const auto& poly : polygons) {
      points.clear();
      for (const auto& vertex : poly.points()) {
        points.emplace_back(vertex.x(), vertex.y());
      }
      if (points.size() < 3) {
        continue;  // Not a valid polygon
      }
      // Green for INCLUDE, Red for EXCLUDE
      cv::Scalar color =
          (poly.type() == exchange_protocol::PolygonType::INCLUDE)
              ? cv::Scalar(0, 255, 0)   // Green
              : cv::Scalar(0, 0, 255);  // Red

      const cv::Point* points_ptr = points.data();
      int num_points = static_cast<int>(points.size());
      cv::polylines(image, &points_ptr, &num_points, 1, true, color, 2);
    }
  }

  std::shared_ptr<IObjectDetector> detector_;
};

std::atomic<bool> stop_requested(false);

void HandleSignal(int signum) {
  stop_requested.store(true, std::memory_order_release);
}


/**
 * @brief Runs the gRPC server
 *
 * @param server_address Address to bind the server (e.g., "0.0.0.0:50051")
 * @param detector IObjectDetector instance
 */
void RunServer(const std::string& server_address,
               std::shared_ptr<IObjectDetector> detector) {
  ObjectDetectorServiceImpl service(detector);

  grpc::EnableDefaultHealthCheckService(true);

  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  // Set max message size to 100MB for large images
  static constexpr int kMaxMessageSizeMB = 100;
  static constexpr int kMaxMessageSizeBytes = kMaxMessageSizeMB * 1024 * 1024;

  builder.SetMaxReceiveMessageSize(kMaxMessageSizeBytes);
  builder.SetMaxSendMessageSize(kMaxMessageSizeBytes);

  std::unique_ptr<Server> server(builder.BuildAndStart());
  spdlog::info("Server started and listening on {}", server_address);
  spdlog::info("Ready to process detection requests...");

  // Setup signal handlers
  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);

  // Wait until user requests stop
  while (!stop_requested.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  // Shutdown gracefully
  server->Shutdown();
  spdlog::info("Server stopped cleanly.");

}


int main(int argc, char** argv) {

  std::string server_address("0.0.0.0:50051");
  std::string model_config = "data/models/yolov3.cfg";
  std::string model_weights = "data/models/yolov3.weights";
  std::string class_names = "data/models/coco.names";

  cxxopts::Options options("server", "Object Detection gRPC Server");

  options.add_options()
    ("h, help", "Show help")
    ("a, address", "Server address", cxxopts::value<std::string>(server_address))
    ("c, config", "Model config file", cxxopts::value<std::string>(model_config))
    ("w, weights", "Model weights file", cxxopts::value<std::string>(model_weights))
    ("n, names", "Class names file", cxxopts::value<std::string>(class_names));

  try {
    auto result = options.parse(argc, argv);
    if (result.count("help")) {
      std::cout << options.help() << '\n';
      return 0;
    }
  } catch (const cxxopts::OptionException& e) {
    std::cerr << "Error parsing options: " << e.what() << '\n';
    std::cout << "Use --help to see usage." << '\n';
    return 1;
  }


  spdlog::set_level(spdlog::level::info);
  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] %v");


  spdlog::info("Starting Object Detection Service...");
  spdlog::info("Model config file: {}", model_config);
  spdlog::info("Model weights file: {}", model_weights);
  spdlog::info("Class names file: {}", class_names);

  try {
    // Initialize YOLO detector
    auto detector = std::make_shared<YoloDetector>(model_config, model_weights,
                                                   class_names, 0.5f, 0.4f);

    // Run server
    RunServer(server_address, detector);

  } catch (const std::exception& e) {
    spdlog::error("Fatal error: {}", e.what());
    return 1;
  }

  return 0;
}