# Object detection gRPC service with polygon masking


A high-performance gRPC-based object detection service that uses YOLOv3 to detect objects within user-defined polygonal regions. The service supports complex detection scenarios with include/exclude zones, priority-based polygon overlapping, and per-class filtering.





## Features

- **YOLOv3 Object Detection**: Detects 80 object classes from the COCO dataset
- **Flexible Polygon Regions**: Define arbitrary polygonal detection zones (≥3 vertices)
- **Include/Exclude Zones**: Control where objects should be detected
- **Priority System**: Handle overlapping polygons with configurable priorities
- **Class Filtering**: Apply polygon rules to specific object classes
- **gRPC API**: Efficient client-server communication with Protocol Buffers
- **Docker Support**: Easy deployment with containerization

## Architecture

```
┌─────────────┐                  ┌─────────────┐
│   Client    │  ──── gRPC ───>  │   Server    │
│             │                  │             │
│ - Send Image│  <─── gRPC ────  │ - YOLOv3    │
│ - Polygons  │                  │ - Filtering │
│ - Display   │                  │ - Detection │
└─────────────┘                  └─────────────┘
```

### Components

- **YoloDetector**: Performs object detection using OpenCV DNN module
- **PolygonProcessor**: Handles polygon-based filtering with priority logic
- **gRPC Service**: Manages client-server communication
- **Protocol Buffers**: Defines data structures for efficient serialization

## Requirements

- CMake 3.5+
- C++14 or higher
- OpenCV 4.x
- gRPC and Protocol Buffers (automatically fetched)
- Docker (optional, for containerized deployment)

## Project Structure

```
.
├── CMakeLists.txt           # Build configuration
├── Dockerfile               # Container definition
├── README.md                # This file
├── proto/
│   └── exchange_protocol.proto  # gRPC service definition
├── src/
│   ├── server.cpp           # gRPC server implementation
│   ├── client.cpp           # gRPC client implementation
│   ├── yolo_detector.h      # YOLO detector header
│   ├── yolo_detector.cpp    # YOLO detector implementation
│   └── polygon_processor.h  # Polygon filtering logic
├── tests/
│   ├── test_polygon_processor.cpp  # Polygon tests
│   └── test_yolo_detector.cpp      # YOLO tests
├── assets/
│   └── polygons.json        # Example polygon configuration
└── data/
    └── models/              # YOLO model files (auto-downloaded)
        ├── yolov3.cfg
        ├── yolov3.weights
        └── coco.names
```

## Building the Project

### Option 1: Docker Build (Recommended)

```bash
# Build Docker image
docker build -t object-detection-service .

```

### Option 2: Local Build

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y build-essential cmake git wget
sudo apt-get install -y libopencv-dev nlohmann-json3-dev

# Download YOLO model files
mkdir -p data/models
cd data/models
wget https://github.com/pjreddie/darknet/raw/master/cfg/yolov3.cfg
wget https://pjreddie.com/media/files/yolov3.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
cd ../..

# Build project
mkdir build && cd build
cmake ..
make -j$(nproc)

# Run server
./server

# Run client (in another terminal)
./client input.jpg output.jpg localhost:50051 ../assets/polygons.json
```

## Usage

### Starting the Server

```bash
./server [address] [model_config] [model_weights] [class_names]
```

**Parameters:**
- `address`: Server address (default: `0.0.0.0:50051`)
- `model_config`: Path to YOLOv3 config file (default: `data/models/yolov3.cfg`)
- `model_weights`: Path to YOLOv3 weights (default: `data/models/yolov3.weights`)
- `class_names`: Path to class names file (default: `data/models/coco.names`)

**Example:**
```bash
./server 0.0.0.0:50051 data/models/yolov3.cfg data/models/yolov3.weights data/models/coco.names
```
### Starting the Server from Docker container
```bash
# Run server
docker run --rm -it --network host -w /app image-detector-task /app/server
```

### Running the Client

```bash
./client [input_image] [output_image] [server_address] [polygons_json]
```

**Parameters:**
- `input_image`: Path to input image (default: `input.jpg`)
- `output_image`: Path to save result (default: `output.jpg`)
- `server_address`: Server address (default: `localhost:50051`)
- `polygons_json`: Path to polygon configuration (default: `polygons.json`)

**Example:**
```bash
./client test.jpg result.jpg localhost:50051 assets/polygons.json
```

### Running the Client from Docker container
One must have `input.jpg` and `polygons.json` files in current directory in order to run client from docker
To see how client works change dir to `assets` directory.
```bash
cd assets
```

Make sure you have `input.jpg` and `polygons.json` files in current directory


Then execute following command:
```bash
# Execute this command before launching client from Docker
xhost +local:docker
```
```bash
# Run client (in another terminal)
docker run --rm -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --network host -w /app -v .:/app/assets image-detector-task /app/client
```



## Polygon Configuration

Polygons are defined in JSON format. Each polygon specifies detection rules for specific regions.

### Configuration Format

```json
[
  {
    "points": [[x1, y1], [x2, y2], [x3, y3], ...],
    "type": "INCLUDE" | "EXCLUDE",
    "priority": <integer>,
    "class_filters": ["class1", "class2", ...]
  }
]
```

### Parameters

- **points**: Array of [x, y] coordinates defining polygon vertices (minimum 3 points)
- **type**: 
  - `INCLUDE`: Objects inside this polygon will be detected
  - `EXCLUDE`: Objects inside this polygon will be ignored
- **priority**: Integer value (higher priority overrides lower priority in overlapping regions)
- **class_filters**: Array of class names this polygon applies to (empty array = no classes affected)

### Example Configuration

```json
[
  {
    "points": [[100,100], [500,100], [500,400], [100,400]],
    "type": "INCLUDE",
    "priority": 1,
    "class_filters": ["person", "car"]
  },
  {
    "points": [[200,200], [300,200], [300,300], [200,300]],
    "type": "EXCLUDE",
    "priority": 2,
    "class_filters": ["person"]
  }
]
```

This configuration:
1. Detects `person` and `car` objects in the large rectangle (100,100 to 500,400)
2. Excludes `person` detections in the small rectangle (200,200 to 300,300)
3. The exclude zone has higher priority, so it overrides the include zone in the overlap

### Priority Rules

- Higher priority polygons override lower priority polygons in overlapping regions
- When priorities are equal, EXCLUDE takes precedence over INCLUDE
- Points outside all polygons are not detected
- Empty `class_filters` means the polygon doesn't apply to any class

## Supported Object Classes

The service uses YOLOv3 trained on COCO dataset with 80 classes:

```
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat,
dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella,
handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite,
baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle,
wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange,
broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant,
bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone,
microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors,
teddy bear, hair drier, toothbrush
```

## Testing

### Running Unit Tests

```bash
cd build
make test

# Or run tests individually
./tests/test_polygon_processor
./tests/test_yolo_detector

# Verbose output
ctest --verbose

```

### Running Unit Tests from Docker container
```bash
# Run tests
docker run --rm -it -w /app image-detector-task ctest

```

### Test Coverage

The test suite includes:

**PolygonProcessor Tests:**
- Basic INCLUDE/EXCLUDE polygon functionality
- Class filtering logic
- Empty class_filters handling
- Priority-based polygon resolution
- Same-priority EXCLUDE precedence
- Complex overlapping scenarios
- Boundary point handling
- Invalid polygon validation
- Non-convex polygon support
- Multiple non-overlapping polygons

**YoloDetector Tests:**
- Model loading and initialization
- Error handling for invalid files
- Detection with various polygon configurations
- Confidence threshold effects
- NMS threshold configuration
- Detection structure validation

## API Documentation

### gRPC Service Definition

```protobuf
service ObjectDetectorService {
    rpc DetectObjects(DetectionRequest) returns (DetectionResponse);
}

message DetectionRequest {
    bytes image_data = 1;
    repeated PolygonConfig polygons = 2;
}

message DetectionResponse {
    bytes result_image_data = 1;
}
```

### Generate Doxygen Documentation

```bash
cd build
make docs

# Open documentation
xdg-open docs/html/index.html  # Linux
open docs/html/index.html       # macOS
```

### Generate Doxygen Documentation from Docker image
```bash
docker run --rm -it -w /app -v .:/app/docs image-detector-task make docs
```
This command will create `html` subdirectory in the current direcory on host

## Performance Considerations

### Optimization Techniques

1. **Bounding Box Pre-check**: Fast rectangle containment test before expensive polygon test
2. **NMS (Non-Maximum Suppression)**: Eliminates duplicate detections
3. **Confidence Thresholding**: Filters low-confidence detections
4. **Efficient Memory Management**: Move semantics and reserve() calls

### Typical Performance

- **Detection Time**: 100-300ms per frame (640x480, CPU)
- **Network Overhead**: ~5-10ms for typical images
- **Memory Usage**: ~500MB (model loaded)

### Tuning Parameters

Adjust detection sensitivity in server initialization:
```cpp
YoloDetector detector(config, weights, names, 
    0.5f,  // confidence_threshold (0.0-1.0)
    0.4f   // nms_threshold (0.0-1.0)
);
```

## Code Style

This project follows [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html):

- **Naming**: 
  - Classes: `PascalCase`
  - Functions: `PascalCase`
  - Variables: `snake_case`
  - Private members: `snake_case_`
  - Constants: `kPascalCase`
  
- **Formatting**: 
  - 2-space indentation
  - 120-character line limit (flexible)
  - Include guards: `PROJECT_PATH_FILE_H_`

## Commit Convention

Commits follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(polygon): add bounding box optimization for polygon checks
fix(yolo): correct memory leak in detection processing
docs(readme): update build instructions for macOS
test(polygon): add tests for non-convex polygons
```

