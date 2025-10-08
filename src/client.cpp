#include <iostream>
#include <memory>
#include <string>
#include <fstream>

#include <grpcpp/grpcpp.h>
#include <opencv2/opencv.hpp>

#include <proto/exchange_protocol.pb.h>
#include <proto/exchange_protocol.grpc.pb.h>

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using exchange_protocol::ObjectDetectorService;
using exchange_protocol::DetectionRequest;
using exchange_protocol::DetectionResponse;
using exchange_protocol::PolygonConfig;
using exchange_protocol::Point;
using exchange_protocol::PolygonType;

class ObjectDetectorClient {
public:
    ObjectDetectorClient(std::shared_ptr<Channel> channel)
        : stub_(ObjectDetectorService::NewStub(channel)) {}

    bool DetectObjects(const std::string& image_path,
                      const std::vector<PolygonConfig>& polygons,
                      const std::string& output_path) {
        
        // Read and encode image
        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Failed to read image: " << image_path << std::endl;
            return false;
        }
        
        std::cout << "Image loaded: " << image.cols << "x" << image.rows << std::endl;
        
        std::vector<uchar> encoded_image;
        if (!cv::imencode(".jpg", image, encoded_image)) {
            std::cerr << "Failed to encode image" << std::endl;
            return false;
        }
        
        // Prepare request
        DetectionRequest request;
        request.set_image_data(encoded_image.data(), encoded_image.size());
        
        for (const auto& polygon : polygons) {
            *request.add_polygons() = polygon;
        }
        
        std::cout << "Sending request with " << polygons.size() << " polygons..." << std::endl;
        
        // Call RPC
        DetectionResponse response;
        ClientContext context;
        
        Status status = stub_->DetectObjects(&context, request, &response);
        
        if (!status.ok()) {
            std::cerr << "RPC failed: " << status.error_code() << ": " 
                     << status.error_message() << std::endl;
            return false;
        }
        
        std::cout << "Response received successfully" << std::endl;
        
        // Decode and save result image
        std::vector<uchar> result_data(response.result_image_data().begin(),
                                       response.result_image_data().end());
        cv::Mat result_image = cv::imdecode(result_data, cv::IMREAD_COLOR);
        
        if (result_image.empty()) {
            std::cerr << "Failed to decode result image" << std::endl;
            return false;
        }
        
        if (!cv::imwrite(output_path, result_image)) {
            std::cerr << "Failed to save result image to: " << output_path << std::endl;
            return false;
        }
        
        std::cout << "Result saved to: " << output_path << std::endl;
        
        return true;
    }

private:
    std::unique_ptr<ObjectDetectorService::Stub> stub_;
};

PolygonConfig CreatePolygon(const std::vector<std::pair<int, int>>& points,
                           PolygonType type,
                           int priority,
                           const std::vector<std::string>& class_filters = {}) {
    PolygonConfig polygon;
    
    for (const auto& p : points) {
        Point* point = polygon.add_points();
        point->set_x(p.first);
        point->set_y(p.second);
    }
    
    polygon.set_type(type);
    polygon.set_priority(priority);
    
    for (const auto& filter : class_filters) {
        polygon.add_class_filters(filter);
    }
    
    return polygon;
}

int main(int argc, char** argv) {
    std::string server_address = "localhost:50051";
    std::string image_path = "input.jpg";
    std::string output_path = "output.jpg";
    
    if (argc > 1) {
        image_path = argv[1];
    }
    if (argc > 2) {
        output_path = argv[2];
    }
    if (argc > 3) {
        server_address = argv[3];
    }
    
    std::cout << "Connecting to server: " << server_address << std::endl;
    std::cout << "Input image: " << image_path << std::endl;
    std::cout << "Output image: " << output_path << std::endl;
    
    // Create client
    ObjectDetectorClient client(
        grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials())
    );
    
    // Example: Create some test polygons
    std::vector<PolygonConfig> polygons;
    
    // INCLUDE polygon - detect objects inside this area
    polygons.push_back(CreatePolygon(
        {{100, 100}, {500, 100}, {500, 400}, {100, 400}},
        PolygonType::INCLUDE,
        1,
        {"person", "car"}
    ));
    
    // EXCLUDE polygon - ignore objects in this area
    polygons.push_back(CreatePolygon(
        {{200, 200}, {300, 200}, {300, 300}, {200, 300}},
        PolygonType::EXCLUDE,
        2,
        {"person"}
    ));
    
    // Call detection service
    if (client.DetectObjects(image_path, polygons, output_path)) {
        std::cout << "Detection completed successfully!" << std::endl;
        return 0;
    } else {
        std::cerr << "Detection failed!" << std::endl;
        return 1;
    }
}