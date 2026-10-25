#ifndef VIDEO_INGESTOR_HPP
#define VIDEO_INGESTOR_HPP

#include "frame_queue.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <atomic>
#include <memory>

/**
 * @brief Multithreaded Video Ingestion Backend in C++.
 * Captures multispectral/terrain video streams asynchronously and pushes raw frames
 * into isolated memory queues to eliminate I/O bottlenecks.
 */
class VideoIngestor {
private:
    std::string source_path_;
    int camera_index_;
    bool is_camera_;
    
    std::atomic<bool> is_running_{false};
    std::thread ingestion_thread_;
    
    IsolatedMemoryQueue<cv::Mat> frame_queue_;
    
    std::atomic<uint64_t> total_frames_ingested_{0};
    std::atomic<double> current_fps_{0.0};

    void ingestionLoop();

public:
    VideoIngestor(size_t queue_capacity = 30);
    ~VideoIngestor();

    bool openSource(const std::string& source_path);
    bool openCamera(int camera_index = 0);
    
    void start();
    void stop();
    
    bool getNextFrame(cv::Mat& frame, int timeout_ms = 50);
    
    size_t getQueueSize() const { return frame_queue_.size(); }
    size_t getQueueCapacity() const { return frame_queue_.capacity(); }
    uint64_t getTotalFramesIngested() const { return total_frames_ingested_.load(); }
    double getCurrentFPS() const { return current_fps_.load(); }
    bool isRunning() const { return is_running_.load(); }
};

#endif // VIDEO_INGESTOR_HPP
