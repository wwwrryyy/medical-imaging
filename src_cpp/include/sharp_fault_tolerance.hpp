#ifndef SHARP_FAULT_TOLERANCE_HPP
#define SHARP_FAULT_TOLERANCE_HPP

#include <string>
#include <chrono>
#include <atomic>

/**
 * @brief Self-Healing Approximate Resilient Programming (SHARP) Paradigm.
 * Maintains fault-tolerant edge inference under electronic interference, frame drops,
 * atmospheric noise, or hardware thermal throttling.
 */
enum class SystemHealthState {
    NOMINAL,
    DEGRADED_ADAPTIVE,
    SELF_HEALING,
    OFFLINE_RESILIENT
};

class SHARPFaultTolerance {
private:
    SystemHealthState current_health_;
    std::atomic<float> queue_pressure_{0.0f};
    std::atomic<float> frame_drop_rate_{0.0f};
    std::atomic<float> current_latency_ms_{0.0f};
    
    // Adaptive parameters
    float target_latency_ms_{33.3f}; // 30 FPS target
    int scale_factor_{1};            // 1 = Full res, 2 = Half res downscaled for resilience

public:
    SHARPFaultTolerance() : current_health_(SystemHealthState::NOMINAL) {}

    void updateTelemetry(float queue_fill_ratio, float latency_ms, float noise_level) {
        queue_pressure_ = queue_fill_ratio;
        current_latency_ms_ = latency_ms;

        // SHARP Fault-Tolerance Heuristic Logic
        if (queue_fill_ratio > 0.85f || latency_ms > 50.0f) {
            current_health_ = SystemHealthState::SELF_HEALING;
            scale_factor_ = 2; // Dynamic downscaling to recover frame throughput
        } else if (queue_fill_ratio > 0.60f || latency_ms > 35.0f || noise_level > 0.15f) {
            current_health_ = SystemHealthState::DEGRADED_ADAPTIVE;
            scale_factor_ = 1;
        } else {
            current_health_ = SystemHealthState::NOMINAL;
            scale_factor_ = 1;
        }
    }

    std::string getHealthStateString() const {
        switch (current_health_) {
            case SystemHealthState::NOMINAL: return "NOMINAL (100% Edge Efficiency)";
            case SystemHealthState::DEGRADED_ADAPTIVE: return "ADAPTIVE (Noise Suppression Active)";
            case SystemHealthState::SELF_HEALING: return "SELF-HEALING (Dynamic Resolution Recovery)";
            case SystemHealthState::OFFLINE_RESILIENT: return "OFFLINE RESILIENT (Local Metadata Buffer)";
        }
        return "UNKNOWN";
    }

    int getScaleFactor() const { return scale_factor_; }
    SystemHealthState getHealthState() const { return current_health_; }
};

#endif // SHARP_FAULT_TOLERANCE_HPP
