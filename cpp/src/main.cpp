#include "fineloc/fineloc.hpp"

#include <array>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string readText(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open: " + path);
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

std::vector<float> jsonArray(const std::string& json, const std::string& key) {
    const std::string quoted = "\"" + key + "\"";
    std::size_t position = json.find(quoted);
    if (position == std::string::npos) return {};
    position = json.find('[', position + quoted.size());
    if (position == std::string::npos) return {};
    const std::size_t end = json.find(']', position + 1);
    if (end == std::string::npos) return {};
    std::vector<float> values;
    const char* cursor = json.c_str() + position + 1;
    const char* finish = json.c_str() + end;
    while (cursor < finish) {
        while (cursor < finish && (std::isspace(static_cast<unsigned char>(*cursor)) || *cursor == ',')) ++cursor;
        if (cursor >= finish) break;
        char* number_end = nullptr;
        const float value = std::strtof(cursor, &number_end);
        if (number_end == cursor) throw std::runtime_error("invalid numeric array for JSON key: " + key);
        values.push_back(value);
        cursor = number_end;
    }
    return values;
}

fineloc::Vec3 vec3FromJson(const std::string& json, const std::string& key, bool required) {
    const std::vector<float> values = jsonArray(json, key);
    if (values.size() < 3) {
        if (required) throw std::runtime_error("missing JSON vec3: " + key);
        return {};
    }
    return {values[0], values[1], values[2]};
}

std::vector<fineloc::Vec3> readBinaryXyzPcd(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open PCD: " + path);
    std::string line;
    std::size_t point_count = 0;
    bool binary = false;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream fields(line);
        std::string key;
        fields >> key;
        if (key == "POINTS") fields >> point_count;
        if (key == "DATA") {
            std::string mode;
            fields >> mode;
            binary = mode == "binary";
            break;
        }
    }
    if (!binary || point_count == 0) throw std::runtime_error("only binary XYZ PCD is supported: " + path);
    std::vector<std::array<float, 3>> raw(point_count);
    stream.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size() * sizeof(raw[0])));
    if (stream.gcount() != static_cast<std::streamsize>(raw.size() * sizeof(raw[0]))) throw std::runtime_error("truncated PCD: " + path);
    std::vector<fineloc::Vec3> points;
    points.reserve(point_count);
    for (const auto& point : raw) {
        if (std::isfinite(point[0]) && std::isfinite(point[1]) && std::isfinite(point[2])) points.push_back({point[0], point[1], point[2]});
    }
    return points;
}

void printVec(const fineloc::Vec3& value) {
    std::cout << '[' << value.x << ',' << value.y << ',' << value.z << ']';
}

std::string argument(int argc, char** argv, const std::string& name) {
    for (int index = 1; index + 1 < argc; ++index) {
        if (argv[index] == name) return argv[index + 1];
    }
    throw std::runtime_error("missing argument: " + name);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 9) {
            std::cerr << "Usage: fineloc_cli --cloud cloud.pcd --param param.txt --stage1 stage1.onnx --stage2 stage2.onnx\n";
            return 2;
        }
        const std::string cloud_path = argument(argc, argv, "--cloud");
        const std::string param_path = argument(argc, argv, "--param");
        const std::string stage1_path = argument(argc, argv, "--stage1");
        const std::string stage2_path = argument(argc, argv, "--stage2");

        const std::string json = readText(param_path);
        fineloc::GeometryPrior prior;
        prior.reference_point = vec3FromJson(json, "startPos", true);
        prior.seam_end_point = vec3FromJson(json, "endPos1", true);
        for (int index = 0; index < 3; ++index) {
            prior.candidate_plane_normals[static_cast<std::size_t>(index)] =
                vec3FromJson(json, "normalPlane" + std::to_string(index + 1), false);
        }

        const std::vector<fineloc::Vec3> points = readBinaryXyzPcd(cloud_path);
        fineloc::FineLocationEngine engine(stage1_path, stage2_path);
        const fineloc::Prediction result = engine.infer(points, prior);

        std::cout << std::fixed << std::setprecision(7);
        std::cout << "{\n  \"point_count\": " << points.size() << ",\n  \"coarse_point\": "; printVec(result.coarse_point);
        std::cout << ",\n  \"final_start_point\": "; printVec(result.final_start_point);
        std::cout << ",\n  \"final_line_direction\": "; printVec(result.final_line_direction);
        std::cout << ",\n  \"final_plane_normals\": ["; printVec(result.final_plane_normals[0]); std::cout << ','; printVec(result.final_plane_normals[1]);
        std::cout << "],\n  \"selected_plane_indices\": [" << result.selected_plane_indices[0] << ',' << result.selected_plane_indices[1] << ']';
        std::cout << ",\n  \"selected_plane_intersection_angle_deg\": " << result.selected_plane_intersection_angle_deg;
        std::cout << ",\n  \"plane_basis_determinant\": " << result.plane_basis_determinant;
        std::cout << ",\n  \"stage2_knn_radius_mm\": " << result.stage2_knn_radius_mm;
        std::cout << ",\n  \"inference_seconds\": " << result.inference_seconds << "\n}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FineLocation error: " << error.what() << '\n';
        return 1;
    }
}
