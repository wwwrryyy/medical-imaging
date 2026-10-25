#ifndef FRAME_QUEUE_HPP
#ifndef FRAME_QUEUE_HPP
#define FRAME_QUEUE_HPP

#include <queue>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <chrono>
#include <atomic>

/**
 * @brief Thread-safe Bounded Memory Queue for raw video frames.
 * Prevents frame-dropping and memory bottlenecks during high-framerate live ingestion.
 */
template <typename T>
class IsolatedMemoryQueue {
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_push_;
    std::condition_variable cv_pop_;
    size_t capacity_;
    std::atomic<bool> stopped_{false};

public:
    explicit IsolatedMemoryQueue(size_t capacity = 30) : capacity_(capacity) {}

    bool push(const T& item, int timeout_ms = 10) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_push_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                               [this] { return queue_.size() < capacity_ || stopped_; })) {
            // Queue full: overwrite oldest frame to guarantee ultra-low latency streaming
            if (!queue_.empty()) {
                queue_.pop();
            }
        }

        if (stopped_) return false;

        queue_.push(item);
        lock.unlock();
        cv_pop_.notify_one();
        return true;
    }

    bool pop(T& item, int timeout_ms = 50) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_pop_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                              [this] { return !queue_.empty() || stopped_; })) {
            return false;
        }

        if (stopped_ && queue_.empty()) return false;

        item = std::move(queue_.front());
        queue_.pop();
        lock.unlock();
        cv_push_.notify_one();
        return true;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    size_t capacity() const {
        return capacity_;
    }

    void stop() {
        stopped_ = true;
        cv_push_.notify_all();
        cv_pop_.notify_all();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::queue<T> empty;
        std::swap(queue_, empty);
    }
};

#endif // FRAME_QUEUE_HPP
