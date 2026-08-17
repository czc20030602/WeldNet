#include "fineloc/fineloc.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace fineloc {
namespace {

constexpr float kEpsilon = 1.0e-6F;
constexpr float kPi = 3.14159265358979323846F;

Vec3 operator+(const Vec3& a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator*(const Vec3& a, float scale) { return {a.x * scale, a.y * scale, a.z * scale}; }

float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
float squaredNorm(const Vec3& value) { return dot(value, value); }
float norm(const Vec3& value) { return std::sqrt(squaredNorm(value)); }
Vec3 normalize(const Vec3& value) {
    const float length = norm(value);
    return length < kEpsilon ? Vec3{} : value * (1.0F / length);
}
Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

struct Mat3 {
    std::array<float, 9> values{};
    float& at(int row, int col) { return values[static_cast<std::size_t>(row * 3 + col)]; }
    float at(int row, int col) const { return values[static_cast<std::size_t>(row * 3 + col)]; }
};

Mat3 identityMatrix() {
    Mat3 matrix;
    matrix.at(0, 0) = matrix.at(1, 1) = matrix.at(2, 2) = 1.0F;
    return matrix;
}

Mat3 transpose(const Mat3& matrix) {
    Mat3 output;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) output.at(row, col) = matrix.at(col, row);
    }
    return output;
}

float determinant(const Mat3& m) {
    return m.at(0, 0) * (m.at(1, 1) * m.at(2, 2) - m.at(1, 2) * m.at(2, 1))
        - m.at(0, 1) * (m.at(1, 0) * m.at(2, 2) - m.at(1, 2) * m.at(2, 0))
        + m.at(0, 2) * (m.at(1, 0) * m.at(2, 1) - m.at(1, 1) * m.at(2, 0));
}

Mat3 inverse(const Mat3& m) {
    const float det = determinant(m);
    if (!std::isfinite(det) || std::abs(det) < 1.0e-8F) throw std::runtime_error("singular 3x3 matrix");
    Mat3 out;
    out.at(0, 0) = (m.at(1, 1) * m.at(2, 2) - m.at(1, 2) * m.at(2, 1)) / det;
    out.at(0, 1) = (m.at(0, 2) * m.at(2, 1) - m.at(0, 1) * m.at(2, 2)) / det;
    out.at(0, 2) = (m.at(0, 1) * m.at(1, 2) - m.at(0, 2) * m.at(1, 1)) / det;
    out.at(1, 0) = (m.at(1, 2) * m.at(2, 0) - m.at(1, 0) * m.at(2, 2)) / det;
    out.at(1, 1) = (m.at(0, 0) * m.at(2, 2) - m.at(0, 2) * m.at(2, 0)) / det;
    out.at(1, 2) = (m.at(0, 2) * m.at(1, 0) - m.at(0, 0) * m.at(1, 2)) / det;
    out.at(2, 0) = (m.at(1, 0) * m.at(2, 1) - m.at(1, 1) * m.at(2, 0)) / det;
    out.at(2, 1) = (m.at(0, 1) * m.at(2, 0) - m.at(0, 0) * m.at(2, 1)) / det;
    out.at(2, 2) = (m.at(0, 0) * m.at(1, 1) - m.at(0, 1) * m.at(1, 0)) / det;
    return out;
}

Vec3 rowMultiply(const Vec3& value, const Mat3& matrix) {
    return {
        value.x * matrix.at(0, 0) + value.y * matrix.at(1, 0) + value.z * matrix.at(2, 0),
        value.x * matrix.at(0, 1) + value.y * matrix.at(1, 1) + value.z * matrix.at(2, 1),
        value.x * matrix.at(0, 2) + value.y * matrix.at(1, 2) + value.z * matrix.at(2, 2),
    };
}

Mat3 matrixFromColumns(const Vec3& first, const Vec3& second, const Vec3& third) {
    Mat3 matrix;
    matrix.at(0, 0) = first.x; matrix.at(1, 0) = first.y; matrix.at(2, 0) = first.z;
    matrix.at(0, 1) = second.x; matrix.at(1, 1) = second.y; matrix.at(2, 1) = second.z;
    matrix.at(0, 2) = third.x; matrix.at(1, 2) = third.y; matrix.at(2, 2) = third.z;
    return matrix;
}

Mat3 lineFrame(const Vec3& end_delta) {
    const Vec3 x_axis = normalize(end_delta);
    if (norm(x_axis) < kEpsilon) return identityMatrix();
    Vec3 up{0.0F, 0.0F, 1.0F};
    if (std::abs(dot(up, x_axis)) > 0.95F) up = {0.0F, 1.0F, 0.0F};
    Vec3 y_axis = normalize(cross(up, x_axis));
    if (norm(y_axis) < kEpsilon) y_axis = {0.0F, 1.0F, 0.0F};
    const Vec3 z_axis = normalize(cross(x_axis, y_axis));
    return matrixFromColumns(x_axis, y_axis, z_axis);
}

struct BasisInfo {
    Mat3 frame;
    Mat3 inverse_frame;
    std::array<Vec3, 2> selected_normals;
    std::array<int, 2> selected_indices;
    float angle_deg = 0.0F;
    float determinant = 0.0F;
    bool fallback = false;
};

BasisInfo buildBasis(const GeometryPrior& prior) {
    const Vec3 seam_direction = normalize(prior.seam_end_point - prior.reference_point);
    float best_angle = std::numeric_limits<float>::infinity();
    std::array<int, 2> best{-1, -1};
    constexpr std::array<std::array<int, 2>, 3> pairs{{{{0, 1}}, {{0, 2}}, {{1, 2}}}};
    for (const auto& pair : pairs) {
        const Vec3 n1 = normalize(prior.candidate_plane_normals[static_cast<std::size_t>(pair[0])]);
        const Vec3 n2 = normalize(prior.candidate_plane_normals[static_cast<std::size_t>(pair[1])]);
        const Vec3 intersection = normalize(cross(n1, n2));
        if (norm(n1) < kEpsilon || norm(n2) < kEpsilon || norm(intersection) < kEpsilon || norm(seam_direction) < kEpsilon) continue;
        const float cosine = std::clamp(std::abs(dot(intersection, seam_direction)), 0.0F, 1.0F);
        const float angle = std::acos(cosine) * 180.0F / kPi;
        if (angle < best_angle) {
            best_angle = angle;
            best = pair;
        }
    }
    if (best[0] < 0) throw std::invalid_argument("at least two valid plane normals are required");

    BasisInfo info;
    info.selected_indices = {best[0] + 1, best[1] + 1};
    info.selected_normals = {
        normalize(prior.candidate_plane_normals[static_cast<std::size_t>(best[0])]),
        normalize(prior.candidate_plane_normals[static_cast<std::size_t>(best[1])]),
    };
    info.angle_deg = best_angle;
    const Mat3 basis = matrixFromColumns(seam_direction, info.selected_normals[0], info.selected_normals[1]);
    info.determinant = determinant(basis);
    if (!std::isfinite(info.determinant) || std::abs(info.determinant) < 1.0e-4F) {
        info.frame = lineFrame(prior.seam_end_point - prior.reference_point);
        info.fallback = true;
    } else {
        info.frame = inverse(transpose(basis));
    }
    info.inverse_frame = inverse(info.frame);
    return info;
}

struct ModelOutput {
    Vec3 delta;
    Vec3 line;
    std::array<Vec3, 2> planes;
};

#ifdef _WIN32
std::wstring ortPath(const std::string& path) { return std::filesystem::u8path(path).wstring(); }
#else
const char* ortPath(const std::string& path) { return path.c_str(); }
#endif

class OnnxModel {
public:
    OnnxModel(Ort::Env& env, const std::string& path, int threads) : session_(nullptr) {
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (threads > 0) options.SetIntraOpNumThreads(threads);
#ifdef _WIN32
        const std::wstring model_path = ortPath(path);
        session_ = Ort::Session(env, model_path.c_str(), options);
#else
        session_ = Ort::Session(env, ortPath(path), options);
#endif
    }

    ModelOutput run(const std::vector<Vec3>& points) {
        std::vector<float> input;
        input.reserve(points.size() * 3);
        for (const Vec3& point : points) {
            input.push_back(point.x); input.push_back(point.y); input.push_back(point.z);
        }
        const std::array<int64_t, 3> shape{1, static_cast<int64_t>(points.size()), 3};
        Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value tensor = Ort::Value::CreateTensor<float>(memory, input.data(), input.size(), shape.data(), shape.size());
        const std::array<const char*, 1> input_names{"points"};
        const std::array<const char*, 3> output_names{"pred_delta", "pred_line_dir", "pred_plane_normals"};
        auto outputs = session_.Run(
            Ort::RunOptions{nullptr}, input_names.data(), &tensor, 1, output_names.data(), output_names.size());
        const float* delta = outputs[0].GetTensorData<float>();
        const float* line = outputs[1].GetTensorData<float>();
        const float* planes = outputs[2].GetTensorData<float>();
        return {
            {delta[0], delta[1], delta[2]},
            normalize({line[0], line[1], line[2]}),
            {{{planes[0], planes[1], planes[2]}, {planes[3], planes[4], planes[5]}}},
        };
    }

private:
    Ort::Session session_;
};

std::vector<Vec3> randomSample(const std::vector<Vec3>& points, std::size_t count, std::uint32_t seed) {
    if (points.size() < count) throw std::invalid_argument("raw point count is smaller than stage-1 sample count");
    std::vector<std::size_t> indices(points.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 generator(seed);
    std::shuffle(indices.begin(), indices.end(), generator);
    std::vector<Vec3> result;
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index) result.push_back(points[indices[index]]);
    return result;
}

std::pair<std::vector<Vec3>, float> knnCrop(
    const std::vector<Vec3>& raw_points,
    const Vec3& reference,
    const Vec3& center_delta,
    std::size_t count) {
    if (raw_points.empty()) throw std::invalid_argument("raw point cloud is empty");
    std::vector<std::pair<float, std::size_t>> distances;
    distances.reserve(raw_points.size());
    for (std::size_t index = 0; index < raw_points.size(); ++index) {
        const Vec3 diff = (raw_points[index] - reference) - center_delta;
        distances.emplace_back(squaredNorm(diff), index);
    }
    const std::size_t kept = std::min(count, distances.size());
    std::nth_element(distances.begin(), distances.begin() + static_cast<std::ptrdiff_t>(kept - 1), distances.end());
    float max_distance2 = 0.0F;
    std::vector<Vec3> crop;
    crop.reserve(count);
    for (std::size_t index = 0; index < kept; ++index) {
        max_distance2 = std::max(max_distance2, distances[index].first);
        crop.push_back((raw_points[distances[index].second] - reference) - center_delta);
    }
    for (std::size_t index = kept; index < count; ++index) crop.push_back(crop[index % kept]);
    return {crop, std::sqrt(max_distance2)};
}

std::vector<Vec3> transformPoints(const std::vector<Vec3>& points, const Mat3& frame) {
    std::vector<Vec3> transformed;
    transformed.reserve(points.size());
    for (const Vec3& point : points) transformed.push_back(rowMultiply(point, frame));
    return transformed;
}

ModelOutput fromBasis(const ModelOutput& local, const BasisInfo& basis) {
    ModelOutput output;
    output.delta = rowMultiply(local.delta, basis.inverse_frame);
    output.line = normalize(rowMultiply(local.line, basis.inverse_frame));
    const Mat3 frame_transpose = transpose(basis.frame);
    output.planes[0] = normalize(rowMultiply(local.planes[0], frame_transpose));
    output.planes[1] = normalize(rowMultiply(local.planes[1], frame_transpose));
    return output;
}

}  // namespace

class FineLocationEngine::Impl {
public:
    Impl(const std::string& stage1, const std::string& stage2, std::size_t n1, std::size_t n2, std::uint32_t seed, int threads)
        : env(ORT_LOGGING_LEVEL_WARNING, "FineLocation"),
          stage1_model(env, stage1, threads),
          stage2_model(env, stage2, threads),
          stage1_points(n1),
          stage2_points(n2),
          sampling_seed(seed) {}

    Ort::Env env;
    OnnxModel stage1_model;
    OnnxModel stage2_model;
    std::size_t stage1_points;
    std::size_t stage2_points;
    std::uint32_t sampling_seed;
};

FineLocationEngine::FineLocationEngine(
    const std::string& stage1_model,
    const std::string& stage2_model,
    std::size_t stage1_points,
    std::size_t stage2_points,
    std::uint32_t sampling_seed,
    int intra_op_threads)
    : impl_(std::make_unique<Impl>(stage1_model, stage2_model, stage1_points, stage2_points, sampling_seed, intra_op_threads)) {}

FineLocationEngine::~FineLocationEngine() = default;
FineLocationEngine::FineLocationEngine(FineLocationEngine&&) noexcept = default;
FineLocationEngine& FineLocationEngine::operator=(FineLocationEngine&&) noexcept = default;

Prediction FineLocationEngine::infer(const std::vector<Vec3>& raw_points, const GeometryPrior& prior) const {
    const auto started = std::chrono::steady_clock::now();
    const BasisInfo basis = buildBasis(prior);

    std::vector<Vec3> stage1 = randomSample(raw_points, impl_->stage1_points, impl_->sampling_seed);
    for (Vec3& point : stage1) point = point - prior.reference_point;
    const ModelOutput coarse = fromBasis(impl_->stage1_model.run(transformPoints(stage1, basis.frame)), basis);

    auto [stage2, radius] = knnCrop(raw_points, prior.reference_point, coarse.delta, impl_->stage2_points);
    const ModelOutput residual = fromBasis(impl_->stage2_model.run(transformPoints(stage2, basis.frame)), basis);

    Prediction prediction;
    prediction.coarse_point = prior.reference_point + coarse.delta;
    prediction.final_start_point = prediction.coarse_point + residual.delta;
    prediction.final_line_direction = residual.line;
    prediction.final_plane_normals = residual.planes;
    prediction.selected_plane_indices = basis.selected_indices;
    prediction.selected_plane_intersection_angle_deg = basis.angle_deg;
    prediction.plane_basis_determinant = basis.determinant;
    prediction.plane_basis_fallback = basis.fallback;
    prediction.stage2_knn_radius_mm = radius;
    prediction.inference_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return prediction;
}

}  // namespace fineloc
