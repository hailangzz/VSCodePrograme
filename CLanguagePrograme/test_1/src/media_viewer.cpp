
#include "media_viewer.h"
#include <iostream>

MediaViewer::MediaViewer(const std::string& dir) : directory(dir) {}

void MediaViewer::show(int max_count, Mode mode) {
    int count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (max_count > 0 && count >= max_count) break;
        
        std::string path = entry.path().string();
        std::string ext = entry.path().extension().string();
        
        if (mode == Mode::IMAGE && 
            std::find(IMAGE_EXTS.begin(), IMAGE_EXTS.end(), ext) != IMAGE_EXTS.end()) {
            displayImage(path);
            count++;
        } 
        else if (mode == Mode::VIDEO && 
                 std::find(VIDEO_EXTS.begin(), VIDEO_EXTS.end(), ext) != VIDEO_EXTS.end()) {
            playVideo(path);
            count++;
        }
    }
}

void MediaViewer::displayImage(const std::string& path) {
    cv::Mat img = cv::imread(path);
    if (!img.empty()) {
        cv::imshow("Image Viewer", img);
        cv::waitKey(0);
        std::cout << "Displayed: " << path << std::endl;
    }
}

void MediaViewer::playVideo(const std::string& path) {
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) return;

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame)) break;
        cv::imshow("Video Player", frame);
        if (cv::waitKey(30) == 27) break; // ESC退出
    }
}
