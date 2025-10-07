#include <iostream>
#include <opencv2/opencv.hpp>


#include <proto/exchange_protocol.pb.h>
#include <proto/exchange_protocol.grpc.pb.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    if (argc != 2) {
        cout << "Usage: my_opencv_app <Image_Path>" << endl;
        return -1;
    }

    // Read the image file
    Mat image;
    image = imread(argv[1], IMREAD_COLOR); 

    if (image.empty()) { // Check for invalid input
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    namedWindow("Display Image", WINDOW_AUTOSIZE); // Create a window for display
    imshow("Display Image", image); // Show the image

    waitKey(0); // Wait for a keystroke in the window

    return 0;
}
