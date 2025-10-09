#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <opencv2/opencv.hpp>

#include "proto/exchange_protocol.pb.h"
#include "proto/exchange_protocol.grpc.pb.h"
#include "yolo_detector.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using grpc::StatusCode;

using exchange_protocol::ObjectDetectorService;
using exchange_protocol::DetectionRequest;
using exchange_protocol::DetectionResponse;

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
    explicit ObjectDetectorServiceImpl(std::shared_ptr<YoloDetector> detector)
        : detector_(std::move(detector)) {}

    /**
     * @brief Handles object detection RPC requests
     * 
     * @param context gRPC server context
     * @param request Detection request with image and polygons
     * @param response Detection response with result image
     * @return gRPC status
     */
    Status DetectObjects(ServerContext* context,
                        const DetectionRequest* request,
                        DetectionResponse* response) override {
    
        std::cout << "=== Received DetectObjects request === \n";
        
        try {
            // Decode image from request
            std::vector<uchar> image_data(request->image_data().begin(), 
                                        request->image_data().end());
            cv::Mat image = cv::imdecode(image_data, cv::IMREAD_COLOR);
            
            if (image.empty()) {
                return Status(StatusCode::INVALID_ARGUMENT, 
                            "Failed to decode input image");
            }
            
            std::cout << "Image decoded: " << image.cols << "x" << image.rows << '\n';
            

            std::vector<exchange_protocol::PolygonConfig> polygons(
                request->polygons().begin(), request->polygons().end());

            // Perform object detection
            std::vector<Detection> detections = detector_->Detect(image, polygons);
            std::cout << "Detected " << detections.size() << " objects\n";
            
            DrawDetections(image, detections);
            DrawPolygons(image, polygons);

            // Encode result image
            std::vector<uchar> encoded_image;
            std::vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY, 95};
            
            if (!cv::imencode(".jpg", image, encoded_image, encode_params)) {
                return Status(StatusCode::INTERNAL, "Failed to encode result image");
            }
            
            response->set_result_image_data(encoded_image.data(), encoded_image.size());
            
            std::cout << "Detection completed successfully. Returning " 
                      << detections.size() << " objects\n";

            return Status::OK;
            
        } catch (const std::exception& e) {
                std::cerr << "Error during detection: " << e.what() << '\n';
                return Status(StatusCode::INTERNAL, std::string("Detection error: ") + e.what());
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
            std::string label = det.class_name + " " + 
                                std::to_string(static_cast<int>(det.confidence * 100)) + 
                                "%";
            
            // Draw label background
            int baseline;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                                  0.5, 1, &baseline);
            
            int label_y = std::max(det.bbox.y, label_size.height);
            cv::rectangle(image,
                          cv::Point(det.bbox.x, label_y - label_size.height),
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
    void DrawPolygons(cv::Mat& image, const std::vector<exchange_protocol::PolygonConfig>& polygons) const {
        for (const auto& poly : polygons) {
            std::vector<cv::Point> points;
            for (const auto& vertex : poly.points()) {
                points.emplace_back(vertex.x(), vertex.y());
            }
            if (points.size() < 3) {
                continue; // Not a valid polygon
            }
            // Green for INCLUDE, Red for EXCLUDE
            cv::Scalar color = (poly.type() == exchange_protocol::PolygonType::INCLUDE)
                ? cv::Scalar(0, 255, 0)   // Green
                : cv::Scalar(0, 0, 255);  // Red

            const cv::Point* pts = points.data();
            int npts = static_cast<int>(points.size());
            cv::polylines(image, &pts, &npts, 1, true, color, 2);
        }
    }

    std::shared_ptr<YoloDetector> detector_;  ///< YOLO detector instance
};

/**
 * @brief Runs the gRPC server
 * 
 * @param server_address Address to bind the server (e.g., "0.0.0.0:50051")
 * @param detector YOLO detector instance
 */
void RunServer(const std::string& server_address,
               std::shared_ptr<YoloDetector> detector) {
    ObjectDetectorServiceImpl service(detector);
    
    grpc::EnableDefaultHealthCheckService(true);
    
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    
    // Set max message size to 100MB for large images
    builder.SetMaxReceiveMessageSize(100 * 1024 * 1024);
    builder.SetMaxSendMessageSize(100 * 1024 * 1024);
    
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << '\n';
    std::cout << "Ready to process detection requests...\n";
    
    server->Wait();
}

int main(int argc, char** argv) {
    std::string server_address("0.0.0.0:50051");
    std::string model_config = "data/models/yolov3.cfg";
    std::string model_weights = "data/models/yolov3.weights";
    std::string class_names = "data/models/coco.names";
    
    // Parse command line arguments
    if (argc > 1) {
        server_address = argv[1];
    }
    if (argc > 2) {
        model_config = argv[2];
    }
    if (argc > 3) {
        model_weights = argv[3];
    }
    if (argc > 4) {
        class_names = argv[4];
    }

    std::cout << "Starting Object Detection Service...\n";
    std::cout << "Model config: " << model_config << '\n';
    std::cout << "Model weights: " << model_weights << '\n';
    std::cout << "Class names: " << class_names << '\n';

    try {
        // Initialize YOLO detector
        auto detector = std::make_shared<YoloDetector>(
            model_config, model_weights, class_names, 0.5f, 0.4f);
        
        // Run server
        RunServer(server_address, detector);
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << '\n';
        return 1;
    }
    
    return 0;
}