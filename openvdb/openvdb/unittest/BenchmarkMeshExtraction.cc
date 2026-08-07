// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// Benchmark: marchingTetrahedra vs volumeToMesh
//
// Usage: vdb_bench_mesh [--runs N] [--radii r0,r1,...] [--voxel v]

#include <openvdb/tools/MarchingTetrahedra.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/util/CpuTimer.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace openvdb;

// ---------------------------------------------------------------------------
// Helpers

static size_t triCountFromQuads(const std::vector<Vec4I>& quads)
{
    return quads.size() * 2; // each quad -> 2 triangles
}

struct RunResult {
    double   ms;          // wall time in ms
    size_t   points;
    size_t   polys;       // triangles (or triangle-equivalent for quads)
};

// Run fn() `runs` times; return per-run results.
template<typename Fn>
static std::vector<RunResult> bench(int runs, Fn fn)
{
    std::vector<RunResult> results;
    results.reserve(runs);
    util::CpuTimer timer;
    for (int i = 0; i < runs; ++i) {
        timer.start();
        RunResult r = fn();
        r.ms = timer.milliseconds();
        results.push_back(r);
    }
    return results;
}

static double mean(const std::vector<RunResult>& v)
{
    double s = 0.0;
    for (const auto& r : v) s += r.ms;
    return s / double(v.size());
}

static double median(std::vector<RunResult> v)
{
    std::sort(v.begin(), v.end(), [](const RunResult& a, const RunResult& b){
        return a.ms < b.ms;
    });
    const size_t n = v.size();
    return n % 2 == 0 ? 0.5 * (v[n/2-1].ms + v[n/2].ms) : v[n/2].ms;
}

// ---------------------------------------------------------------------------

static void printRow(const std::string& label,
                     const std::vector<RunResult>& results,
                     std::ostream& os)
{
    const double mn = mean(results);
    const double med = median(results);
    const size_t pts = results.back().points;
    const size_t pol = results.back().polys;

    os << "  " << std::left << std::setw(32) << label
       << std::right
       << std::setw(10) << std::fixed << std::setprecision(2) << mn
       << " ms (median " << std::setw(8) << std::fixed << std::setprecision(2) << med
       << " ms)   pts=" << std::setw(8) << pts
       << "  polys=" << std::setw(8) << pol
       << "\n";
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    int runs = 5;
    float voxelSize = 0.5f;
    std::vector<float> radii = {5.0f, 10.0f, 20.0f, 40.0f};

    // Minimal arg parsing
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--runs") == 0 && i+1 < argc) {
            runs = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--voxel") == 0 && i+1 < argc) {
            voxelSize = static_cast<float>(std::atof(argv[++i]));
        } else if (std::strcmp(argv[i], "--radii") == 0 && i+1 < argc) {
            radii.clear();
            std::istringstream ss(argv[++i]);
            std::string tok;
            while (std::getline(ss, tok, ','))
                radii.push_back(static_cast<float>(std::atof(tok.c_str())));
        }
    }

    openvdb::initialize();

    std::cout << "Mesh extraction benchmark\n"
              << "  runs=" << runs << "  voxelSize=" << voxelSize << "\n\n";

    std::cout << std::left  << std::setw(34) << "Method"
              << std::right << std::setw(10) << "mean"
              << "            median"
              << "      points    polys\n";
    std::cout << std::string(80, '-') << "\n";

    for (float radius : radii) {
        FloatGrid::Ptr grid =
            tools::createLevelSetSphere<FloatGrid>(radius, Vec3f(0.0f), voxelSize, 3.0f);

        const Index64 activeVoxels = grid->activeVoxelCount();
        std::cout << "\nradius=" << radius << "  voxelSize=" << voxelSize
                  << "  activeVoxels=" << activeVoxels << "\n";

        // --- marchingTetrahedra ---
        auto resultsMT = bench(runs, [&]() -> RunResult {
            std::vector<Vec3s> pts;
            std::vector<Vec3I> tris;
            tools::marchingTetrahedra(*grid, pts, tris, 0.0);
            return {0.0, pts.size(), tris.size()};
        });
        printRow("marchingTetrahedra", resultsMT, std::cout);

        // --- volumeToMesh (quads only, no adaptivity) ---
        auto resultsV2M = bench(runs, [&]() -> RunResult {
            std::vector<Vec3s> pts;
            std::vector<Vec4I> quads;
            tools::volumeToMesh(*grid, pts, quads, 0.0);
            return {0.0, pts.size(), triCountFromQuads(quads)};
        });
        printRow("volumeToMesh (quads)", resultsV2M, std::cout);

        // --- volumeToMesh (tris+quads, no adaptivity) ---
        auto resultsV2Ma = bench(runs, [&]() -> RunResult {
            std::vector<Vec3s> pts;
            std::vector<Vec3I> tris;
            std::vector<Vec4I> quads;
            tools::volumeToMesh(*grid, pts, tris, quads, 0.0, 0.0);
            return {0.0, pts.size(), tris.size() + triCountFromQuads(quads)};
        });
        printRow("volumeToMesh (tris+quads)", resultsV2Ma, std::cout);

        // Speedup
        const double speedup = mean(resultsV2M) / mean(resultsMT);
        std::cout << "  speedup marchingTetrahedra vs volumeToMesh(quads): "
                  << std::fixed << std::setprecision(2) << speedup << "x\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
