#ifndef ANOMALY_CALCULUS_HPP
#define ANOMALY_CALCULUS_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

/**
 * @brief Anomaly Calculus Engine (Mathematical Detection)
 * Computes pixel-wise Mean Squared Error (MSE) between raw input frames and SNN reconstructed baselines.
 * Generates spatial anomaly heatmaps flagging camouflaged threats, hidden assets, and disturbed terrain.
 */
class AnomalyCalculus {
private:
    float mse_threshold_;
    float gaussian_sigma_;

public:
    explicit AnomalyCalculus(float mse_threshold = 0.08f, float gaussian_sigma = 1.5f);
    ~AnomalyCalculus() = default;

    /**
     * @brief Computes pixel-wise MSE heatmap matrix.
     * @param raw_frame Input OpenCV frame (BGR or Grayscale, CV_8U or CV_32F).
     * @param recon_frame Reconstructed baseline frame from SNN Autoencoder.
     * @param heatmap_out Output CV_32FC1 floating point MSE matrix.
     * @return Mean MSE score across the entire frame.
     */
    float computePixelwiseMSE(const cv::Mat& raw_frame, const cv::Mat& recon_frame, cv::Mat& heatmap_out);

    /**
     * @brief Generates normalized 8-bit visual heatmap (0-255 uint8) suitable for Python / UI display.
     */
    void generateVisualHeatmap(const cv::Mat& mse_matrix, cv::Mat& visual_heatmap, bool apply_colormap = false);

    /**
     * @brief Computes summary statistics over spatial anomaly regions.
     */
    void getAnomalyStats(const cv::Mat& mse_matrix, float& mean_mse, float& max_mse, cv::Point& max_loc);

    void setThreshold(float th) { mse_threshold_ = th; }
    float getThreshold() const { return mse_threshold_; }
};

#endif // ANOMALY_CALCULUS_HPP
