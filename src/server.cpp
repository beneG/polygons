#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include <opencv2/opencv.hpp>

#include <proto/exchange_protocol.pb.h>
#include <proto/exchange_protocol.grpc.pb.h>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

using exchange_protocol::ObjectDetectorService;
using exchange_protocol::DetectionRequest;
using exchange_protocol::DetectionResponse;
using exchange_protocol::PolygonConfig;
using exchange_protocol::Point;
using exchange_protocol::PolygonType;

class ObjectDetectorServiceImpl final : public ObjectDetectorService::Service {
public:
    Status DetectObjects(ServerContext* context,
                        const DetectionRequest* request,
                        DetectionResponse* response) override {
        std::cout << "DetectObjects() called" << std::endl;
        return Status::OK;
    }
};


void RunServer(const std::string& server_address) {
    ObjectDetectorServiceImpl service;
    
    grpc::EnableDefaultHealthCheckService(true);
    
    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;
    
    server->Wait();
}

int main(int argc, char** argv) {
    std::string server_address("0.0.0.0:50051");
    
    if (argc > 1) {
        server_address = argv[1];
    }
    
    std::cout << "Starting Object Detector Service..." << std::endl;
    RunServer(server_address);
    
    return 0;
}
