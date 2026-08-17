#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fineloc {

struct Vec3 {
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
};

struct GeometryPrior {
    Vec3 reference_point;
    Vec3 seam_end_point;
    std::array<Vec3, 3> candidate_plane_normals{};
};

struct Prediction {
    Vec3 coarse_point;
    Vec3 final_start_point;
    Vec3 final_line_direction;
    std::array<Vec3, 2> final_plane_normals{};
    std::array<int, 2> selected_plane_indices{};
    float selected_plane_intersection_angle_deg = 0.0F;
    float plane_basis_determinant = 0.0F;
    bool plane_basis_fallback = false;
    float stage2_knn_radius_mm = 0.0F;
    double inference_seconds = 0.0;
};

class FineLocationEngine {
public:
    FineLocationEngine(
        const std::string& stage1_model,
        const std::string& stage2_model,
        std::size_t stage1_points = 8192,
        std::size_t stage2_points = 16384,
        std::uint32_t sampling_seed = 42,
        int intra_op_threads = 0);
    ~FineLocationEngine();

    FineLocationEngine(FineLocationEngine&&) noexcept;
    FineLocationEngine& operator=(FineLocationEngine&&) noexcept;
    FineLocationEngine(const FineLocationEngine&) = delete;
    FineLocationEngine& operator=(const FineLocationEngine&) = delete;

    Prediction infer(const std::vector<Vec3>& raw_points, const GeometryPrior& prior) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace fineloc
