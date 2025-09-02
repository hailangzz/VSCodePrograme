
#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class MediaViewer {
public:
    enum class Mode { IMAGE, VIDEO };
    
    MediaViewer(const std::string& dir);
    void show(int max_count = -1, Mode mode = Mode::IMAGE);

private:
    std::string directory;
    const std::vector<std::string> IMAGE_EXTS = {".jpg", ".png", ".jpeg"};
    const std::vector<std::string> VIDEO_EXTS = {".mp4", ".avi", ".mov"};
    
    void displayImage(const std::string& path);
    void playVideo(const std::string& path);
};
