#include <grpcpp/grpcpp.h>
#include <proto/exchange_protocol.grpc.pb.h>
#include <proto/exchange_protocol.pb.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <cxxopts.hpp>


using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using exchange_protocol::DetectionRequest;
using exchange_protocol::DetectionResponse;
using exchange_protocol::ObjectDetectorService;
using exchange_protocol::Point;
using exchange_protocol::PolygonConfig;
using exchange_protocol::PolygonType;

using json = nlohmann::json;

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

    std::cout << "Image loaded: " << image.cols << "x" << image.rows
              << std::endl;

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

    std::cout << "Sending request with " << polygons.size() << " polygons..."
              << std::endl;

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
      std::cerr << "Failed to save result image to: " << output_path
                << std::endl;
      return false;
    }

    std::cout << "Result saved to: " << output_path << std::endl;

    // Display result
    cv::namedWindow("Detection Result", cv::WINDOW_NORMAL);
    cv::imshow("Detection Result", result_image);
    std::cout << "Press any key to close..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();

    return true;
  }

 private:
  std::unique_ptr<ObjectDetectorService::Stub> stub_;
};

PolygonConfig CreatePolygon(
    const std::vector<std::pair<int, int>>& points, PolygonType type,
    int priority, const std::vector<std::string>& class_filters = {}) {
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

// Loads polygons from a JSON file
std::vector<PolygonConfig> LoadPolygonsFromJson(const std::string& filename) {
  std::vector<PolygonConfig> polygons;
  std::ifstream ifs(filename);
  if (!ifs) {
    std::cerr << "Failed to open polygon config file: " << filename
              << std::endl;
    return polygons;
  }
  json json_object;
  ifs >> json_object;
  for (const auto& poly : json_object) {
    std::vector<std::pair<int, int>> points;
    for (const auto& pt : poly["points"]) {
      points.emplace_back(pt[0], pt[1]);
    }
    PolygonType type =
        poly["type"] == "INCLUDE" ? PolygonType::INCLUDE : PolygonType::EXCLUDE;
    int priority = poly.value("priority", 1);
    std::vector<std::string> class_filters;
    if (poly.contains("class_filters")) {
      for (const auto& c : poly["class_filters"]) {
        class_filters.push_back(c);
      }
    }
    polygons.push_back(CreatePolygon(points, type, priority, class_filters));
  }
  return polygons;
}

int main(int argc, char** argv) {
  std::string server_address = "localhost:50051";
  std::string image_path = "assets/input.jpg";
  std::string output_path = "assets/output.jpg";
  std::string polygons_path = "assets/polygons.json";

  cxxopts::Options options("client", "Object Detection gRPC Client");

  options.add_options()
      ("h, help", "Show help")
      ("a, address", "Server address", cxxopts::value<std::string>(server_address))
      ("i, input_image", "Input image file", cxxopts::value<std::string>(image_path))
      ("o, output_image", "Output image file", cxxopts::value<std::string>(output_path))
      ("p, polygons", "Polygon config JSON file", cxxopts::value<std::string>(polygons_path));
  
  try {
    auto result = options.parse(argc, argv);
    if (result.count("help")) {
      std::cout << options.help() << std::endl;
      return 0;
    }
  } catch (const cxxopts::OptionException& e) {
    std::cerr << "Error parsing options: " << e.what() << '\n';
    std::cout << options.help() << std::endl;
    std::cout << "Use --help to see usage." << '\n';
    return 1;
  }

  std::cout << "Connecting to server: " << server_address << std::endl;
  std::cout << "Input image: " << image_path << std::endl;
  std::cout << "Output image: " << output_path << std::endl;

  // Create client
  ObjectDetectorClient client(
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials()));

  std::vector<PolygonConfig> polygons = LoadPolygonsFromJson(polygons_path);

  // Call detection service
  if (client.DetectObjects(image_path, polygons, output_path)) {
    std::cout << "Detection completed successfully!" << std::endl;
    return 0;
  } else {
    std::cerr << "Detection failed!" << std::endl;
    return 1;
  }
}