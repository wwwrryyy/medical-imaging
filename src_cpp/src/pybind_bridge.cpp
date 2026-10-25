#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "video_ingestor.hpp"
#include "anomaly_calculus.hpp"
#include "sharp_fault_tolerance.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

namespace py = pybind11;

/**
 * Helper functions for Zero-Copy Inter-Process Memory Bridge.
 * Wraps C++ cv::Mat pointers directly inside NumPy ndarray buffer views without memory duplication.
 */
py::array_t<uint8_t> mat_to_numpy_zerocopy_u8(cv::Mat& mat) {
    if (mat.empty()) return py::array_t<uint8_t>();
    
    std::vector<ssize_t> shape;
    std::vector<ssize_t> strides;
    
    if (mat.channels() == 1) {
        shape = { static_cast<ssize_t>(mat.rows), static_cast<ssize_t>(mat.cols) };
        strides = { static_cast<ssize_t>(mat.step[0]), static_cast<ssize_t>(mat.step[1]) };
    } else {
        shape = { static_cast<ssize_t>(mat.rows), static_cast<ssize_t>(mat.cols), static_cast<ssize_t>(mat.channels()) };
        strides = { static_cast<ssize_t>(mat.step[0]), static_cast<ssize_t>(mat.step[1]), static_cast<ssize_t>(mat.elemSize1()) };
    }
    
    return py::array_t<uint8_t>(shape, strides, mat.data);
}

py::array_t<float> mat_to_numpy_zerocopy_f32(cv::Mat& mat) {
    if (mat.empty()) return py::array_t<float>();
    
    std::vector<ssize_t> shape = { static_cast<ssize_t>(mat.rows), static_cast<ssize_t>(mat.cols) };
    std::vector<ssize_t> strides = { static_cast<ssize_t>(mat.step[0]), static_cast<ssize_t>(mat.step[1]) };
    
    return py::array_t<float>(shape, strides, reinterpret_cast<float*>(mat.data));
}

cv::Mat numpy_to_mat(py::array_t<uint8_t> input_array) {
    py::buffer_info buf = input_array.request();
    int rows = static_cast<int>(buf.shape[0]);
    int cols = static_cast<int>(buf.shape[1]);
    int channels = (buf.ndim == 3) ? static_cast<int>(buf.shape[2]) : 1;
    int type = (channels == 3) ? CV_8UC3 : CV_8UC1;
    
    return cv::Mat(rows, cols, type, buf.ptr).clone();
}

cv::Mat numpy_float_to_mat(py::array_t<float> input_array) {
    py::buffer_info buf = input_array.request();
    int rows = static_cast<int>(buf.shape[0]);
    int cols = static_cast<int>(buf.shape[1]);
    
    return cv::Mat(rows, cols, CV_32FC1, buf.ptr).clone();
}

/**
 * Unified Battlefield Core Engine exposed to Python.
 */
class BattlefieldEngine {
private:
    VideoIngestor ingestor_;
    AnomalyCalculus anomaly_calc_;
    SHARPFaultTolerance sharp_;
    
    cv::Mat current_raw_frame_;
    cv::Mat current_recon_frame_;
    cv::Mat current_mse_heatmap_;
    cv::Mat current_visual_heatmap_;

public:
    BattlefieldEngine() : ingestor_(30), anomaly_calc_(0.08f) {}

    bool startStream(const std::string& source) {
        if (source == "0" || source == "camera") {
            ingestor_.openCamera(0);
        } else {
            ingestor_.openSource(source);
        }
        ingestor_.start();
        return true;
    }

    void stopStream() {
        ingestor_.stop();
    }

    py::tuple processNextFrame(py::array_t<float> recon_input) {
        cv::Mat raw;
        bool has_frame = ingestor_.getNextFrame(raw, 20);
        if (!has_frame) {
            return py::make_tuple(false, py::none(), py::none(), py::none(), 0.0f);
        }

        current_raw_frame_ = raw;
        
        // Convert numpy recon tensor input back to Mat if supplied, else use raw frame baseline
        if (recon_input.size() > 0) {
            current_recon_frame_ = numpy_float_to_mat(recon_input);
        } else {
            current_recon_frame_ = current_raw_frame_.clone();
        }

        // Compute Pixelwise MSE Heatmap
        float mean_mse = anomaly_calc_.computePixelwiseMSE(current_raw_frame_, current_recon_frame_, current_mse_heatmap_);
        
        // Generate Visual Colormap Heatmap
        anomaly_calc_.generateVisualHeatmap(current_mse_heatmap_, current_visual_heatmap_, true);

        // Telemetry update to SHARP engine
        float queue_fill = static_cast<float>(ingestor_.getQueueSize()) / static_cast<float>(ingestor_.getQueueCapacity());
        sharp_.updateTelemetry(queue_fill, 15.0f, mean_mse);

        // Zero-copy array wrapping for Python
        py::array_t<uint8_t> raw_arr = mat_to_numpy_zerocopy_u8(current_raw_frame_);
        py::array_t<float> mse_arr = mat_to_numpy_zerocopy_f32(current_mse_heatmap_);
        py::array_t<uint8_t> visual_arr = mat_to_numpy_zerocopy_u8(current_visual_heatmap_);

        return py::make_tuple(true, raw_arr, mse_arr, visual_arr, mean_mse);
    }

    std::string getHealthStatus() const {
        return sharp_.getHealthStateString();
    }

    double getFPS() const {
        return ingestor_.getCurrentFPS();
    }

    size_t getQueueSize() const {
        return ingestor_.getQueueSize();
    }
};

PYBIND11_MODULE(battlefield_core, m) {
    m.doc() = "C++ Zero-Copy Hardware Ingestion and Anomaly Calculus Core for Battlefield Object Detection";

    py::class_<BattlefieldEngine>(m, "BattlefieldEngine")
        .def(py::init<>())
        .def("start_stream", &BattlefieldEngine::startStream, py::arg("source"))
        .def("stop_stream", &BattlefieldEngine::stopStream)
        .def("process_next_frame", &BattlefieldEngine::processNextFrame)
        .def("get_health_status", &BattlefieldEngine::getHealthStatus)
        .def("get_fps", &BattlefieldEngine::getFPS)
        .def("get_queue_size", &BattlefieldEngine::getQueueSize);

    py::class_<AnomalyCalculus>(m, "AnomalyCalculus")
        .def(py::init<float, float>(), py::arg("mse_threshold") = 0.08f, py::arg("gaussian_sigma") = 1.5f)
        .def("set_threshold", &AnomalyCalculus::setThreshold)
        .def("get_threshold", &AnomalyCalculus::getThreshold);
}
