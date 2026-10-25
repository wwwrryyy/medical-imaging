#include "anomaly_calculus.hpp"
#include <iostream>

AnomalyCalculus::AnomalyCalculus(float mse_threshold, float gaussian_sigma)
    : mse_threshold_(mse_threshold), gaussian_sigma_(gaussian_sigma) {}

float AnomalyCalculus::computePixelwiseMSE(const cv::Mat& raw_frame, const cv::Mat& recon_frame, cv::Mat& heatmap_out) {
    if (raw_frame.empty() || recon_frame.empty()) {
        return 0.0f;
    }

    cv::Mat raw_float, recon_float;
    
    // Normalize inputs to range [0.0, 1.0] CV_32F
    if (raw_frame.type() != CV_32F && raw_frame.type() != CV_32FC3) {
        raw_frame.convertTo(raw_float, CV_32F, 1.0 / 255.0);
    } else {
        raw_float = raw_frame.clone();
    }

    if (recon_frame.type() != CV_32F && recon_frame.type() != CV_32FC3) {
        recon_frame.convertTo(recon_float, CV_32F, 1.0 / 255.0);
    } else {
        recon_float = recon_frame.clone();
    }

    // Ensure matching spatial resolutions
    if (raw_float.size() != recon_float.size()) {
        cv::resize(recon_float, recon_float, raw_float.size(), 0, 0, cv::INTER_LINEAR);
    }

    // Convert multi-channel inputs to single channel grayscale if necessary
    if (raw_float.channels() > 1) {
        cv::cvtColor(raw_float, raw_float, cv::COLOR_BGR2GRAY);
    }
    if (recon_float.channels() > 1) {
        cv::cvtColor(recon_float, recon_float, cv::COLOR_BGR2GRAY);
    }

    // Compute pixel-wise squared difference matrix: (Raw - Recon)^2
    cv::Mat diff;
    cv::absdiff(raw_float, recon_float, diff);
    cv::multiply(diff, diff, heatmap_out);

    // Apply spatial smoothing filter to suppress isolated pixel noise
    if (gaussian_sigma_ > 0.0f) {
        cv::GaussianBlur(heatmap_out, heatmap_out, cv::Size(5, 5), gaussian_sigma_);
    }

    // Calculate mean MSE value across entire frame
    cv::Scalar mean_scalar = cv::mean(heatmap_out);
    return static_cast<float>(mean_scalar[0]);
}

void AnomalyCalculus::generateVisualHeatmap(const cv::Mat& mse_matrix, cv::Mat& visual_heatmap, bool apply_colormap) {
    if (mse_matrix.empty()) return;

    // Normalize MSE floating point matrix (0.0 - max_val) to uint8 (0 - 255)
    double min_val, max_val;
    cv::minMaxLoc(mse_matrix, &min_val, &max_val);

    double upper_bound = std::max(max_val, 0.25);
    cv::Mat normalized;
    mse_matrix.convertTo(normalized, CV_8U, 255.0 / upper_bound);

    if (apply_colormap) {
        cv::applyColorMap(normalized, visual_heatmap, cv::COLORMAP_JET);
    } else {
        visual_heatmap = normalized;
    }
}

void AnomalyCalculus::getAnomalyStats(const cv::Mat& mse_matrix, float& mean_mse, float& max_mse, cv::Point& max_loc) {
    if (mse_matrix.empty()) {
        mean_mse = 0.0f;
        max_mse = 0.0f;
        max_loc = cv::Point(0, 0);
        return;
    }

    double min_v, max_v;
    cv::minMaxLoc(mse_matrix, &min_v, &max_v, nullptr, &max_loc);
    cv::Scalar avg_s = cv::mean(mse_matrix);

    mean_mse = static_cast<float>(avg_s[0]);
    max_mse = static_cast<float>(max_v);
}
