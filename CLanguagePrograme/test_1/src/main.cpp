
#include "media_viewer.h"
#include <getopt.h>
#include <iostream>

int main(int argc, char* argv[]) {
    std::string dir;
    int max_count = -1;
    MediaViewer::Mode mode = MediaViewer::Mode::IMAGE;

    // 命令行参数解析
    int opt;
    while ((opt = getopt(argc, argv, "d:n:m:")) != -1) {
        switch (opt) {
            case 'd': dir = optarg; break;
            case 'n': max_count = std::stoi(optarg); break;
            case 'm': 
                mode = (std::string(optarg) == "video") ? 
                    MediaViewer::Mode::VIDEO : MediaViewer::Mode::IMAGE;
                break;
            default: 
                std::cerr << "Usage: " << argv[0] 
                          << " -d <directory> [-n max_count] [-m image|video]\n";
                return 1;
        }
    }

    if (dir.empty()) {
        std::cerr << "Directory must be specified with -d\n";
        return 1;
    }

    MediaViewer viewer(dir);
    viewer.show(max_count, mode);
    return 0;
}
