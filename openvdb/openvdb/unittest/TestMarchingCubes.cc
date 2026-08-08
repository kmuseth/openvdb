// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/tools/MarchingCubes.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/Exceptions.h>

#include <gtest/gtest.h>

#include <cmath>
#include <map>
#include <utility>
#include <vector>


class TestMarchingCubes: public ::testing::Test
{
};


namespace {

// Summary statistics used to validate a triangle mesh.
struct MeshStats
{
    size_t boundaryEdges = 0;    // edges used by exactly one triangle (open)
    size_t nonManifoldEdges = 0; // edges used by more than two triangles
    size_t degenerateTris = 0;   // triangles with a repeated vertex index
    long   euler = 0;            // V - E + F
    double signedVolume = 0.0;   // >0 for a closed, outward-oriented surface
};

MeshStats
analyze(const std::vector<openvdb::Vec3s>& points, const std::vector<openvdb::Vec3I>& tris)
{
    MeshStats s;
    std::map<std::pair<openvdb::Index32, openvdb::Index32>, int> edges;
    auto addEdge = [&](openvdb::Index32 a, openvdb::Index32 b) {
        if (a > b) std::swap(a, b);
        ++edges[std::make_pair(a, b)];
    };
    for (const openvdb::Vec3I& t : tris) {
        if (t[0] == t[1] || t[1] == t[2] || t[0] == t[2]) ++s.degenerateTris;
        addEdge(t[0], t[1]); addEdge(t[1], t[2]); addEdge(t[2], t[0]);

        const openvdb::Vec3d p0(points[t[0]]), p1(points[t[1]]), p2(points[t[2]]);
        s.signedVolume += p0.dot(p1.cross(p2)); // 6x tetra volume w.r.t. origin
    }
    s.signedVolume /= 6.0;
    for (const auto& e : edges) {
        if (e.second == 1) ++s.boundaryEdges;
        else if (e.second != 2) ++s.nonManifoldEdges;
    }
    s.euler = long(points.size()) - long(edges.size()) + long(tris.size());
    return s;
}

} // unnamed namespace


// A meshed level-set sphere must be a closed, outward-oriented, genus-0 manifold
// whose vertices lie on the sphere and whose enclosed volume matches the analytic
// value.
TEST_F(TestMarchingCubes, testSphere)
{
    using namespace openvdb;

    const float radius = 10.0f, voxel = 0.5f;
    FloatGrid::Ptr grid =
        tools::createLevelSetSphere<FloatGrid>(radius, Vec3f(0.0f), voxel, 3.0f);

    std::vector<Vec3s> points;
    std::vector<Vec3I> triangles;
    tools::marchingCubes(*grid, points, triangles, 0.0);

    ASSERT_FALSE(points.empty());
    ASSERT_FALSE(triangles.empty());

    // Every vertex lies on the sphere surface (Marching Cubes places crossings on
    // cube edges, so they sit very close to the analytic radius).
    for (const Vec3s& p : points) {
        const double r = std::sqrt(double(p.x())*p.x() + double(p.y())*p.y() + double(p.z())*p.z());
        EXPECT_NEAR(r, radius, voxel);
    }

    const MeshStats s = analyze(points, triangles);

    // Watertight 2-manifold, genus 0.
    EXPECT_EQ(size_t(0), s.boundaryEdges);
    EXPECT_EQ(size_t(0), s.nonManifoldEdges);
    EXPECT_EQ(size_t(0), s.degenerateTris);
    EXPECT_EQ(long(2), s.euler); // V - E + F == 2 for a sphere

    // Consistent outward orientation (positive volume) and correct magnitude.
    const double analytic = 4.0 / 3.0 * openvdb::math::pi<double>() * radius * radius * radius;
    EXPECT_GT(s.signedVolume, 0.0);
    EXPECT_NEAR(s.signedVolume, analytic, 0.02 * analytic); // within 2%
}


// The isovalue selects an offset isosurface (SDF == isovalue), so a positive
// isovalue expands the meshed radius and a negative one shrinks it.
TEST_F(TestMarchingCubes, testIsovalue)
{
    using namespace openvdb;

    const float radius = 10.0f, voxel = 0.5f;
    FloatGrid::Ptr grid =
        tools::createLevelSetSphere<FloatGrid>(radius, Vec3f(0.0f), voxel, 3.0f);

    auto meanRadius = [&](double iso) -> double {
        std::vector<Vec3s> pts; std::vector<Vec3I> tris;
        tools::marchingCubes(*grid, pts, tris, iso);
        EXPECT_FALSE(pts.empty());
        double sum = 0.0;
        for (const Vec3s& p : pts) {
            sum += std::sqrt(double(p.x())*p.x() + double(p.y())*p.y() + double(p.z())*p.z());
        }
        return pts.empty() ? 0.0 : sum / double(pts.size());
    };

    EXPECT_NEAR(meanRadius( 0.0), 10.0, 0.1);
    EXPECT_NEAR(meanRadius( 1.0), 11.0, 0.1); // SDF == +1 -> radius 11
    EXPECT_NEAR(meanRadius(-1.0),  9.0, 0.1); // SDF == -1 -> radius 9
}


// A grid whose values never cross the isovalue produces an empty mesh (and the
// output vectors are cleared).
TEST_F(TestMarchingCubes, testNoCrossing)
{
    using namespace openvdb;

    FloatGrid::Ptr grid =
        tools::createLevelSetSphere<FloatGrid>(10.0f, Vec3f(0.0f), 0.5f, 3.0f);

    // The narrow band only stores |SDF| up to ~halfWidth*voxel = 1.5 world units,
    // so an isovalue well outside the band yields no crossing.
    std::vector<Vec3s> points(5); // pre-fill to confirm they get cleared
    std::vector<Vec3I> triangles(3);
    tools::marchingCubes(*grid, points, triangles, 100.0);

    EXPECT_TRUE(points.empty());
    EXPECT_TRUE(triangles.empty());
}


// The output must be in world space: an off-center, finely-voxelized sphere is
// meshed with vertices on the correct world-space surface.
TEST_F(TestMarchingCubes, testWorldSpace)
{
    using namespace openvdb;

    const Vec3f center(3.0f, -4.0f, 2.0f);
    const float radius = 5.0f;
    FloatGrid::Ptr grid =
        tools::createLevelSetSphere<FloatGrid>(radius, center, 0.25f, 3.0f);

    std::vector<Vec3s> points;
    std::vector<Vec3I> triangles;
    tools::marchingCubes(*grid, points, triangles, 0.0);

    ASSERT_FALSE(points.empty());
    for (const Vec3s& p : points) {
        const double r = (Vec3d(p) - Vec3d(center)).length();
        EXPECT_NEAR(r, radius, 0.25); // within a voxel
    }

    const MeshStats s = analyze(points, triangles);
    EXPECT_EQ(size_t(0), s.boundaryEdges);
    EXPECT_EQ(size_t(0), s.nonManifoldEdges);
    EXPECT_EQ(long(2), s.euler);
}


// Topological guarantee of Marching Cubes 33: a saddle-rich field (a gyroid, which
// has many ambiguous faces and interior tunnels) must still mesh to a watertight,
// 2-manifold surface. The naive Marching Cubes would leave cracks on the ambiguous
// faces; MC33 resolves them with the face and interior tests.
TEST_F(TestMarchingCubes, testGyroidTopology)
{
    using namespace openvdb;

    const int N = 60;
    const double L = 6.0 * openvdb::math::pi<double>();
    const double h = L / N;

    FloatGrid::Ptr grid = FloatGrid::create(3.0f);
    {
        FloatGrid::Accessor acc = grid->getAccessor();
        for (int i = 0; i <= N; ++i)
            for (int j = 0; j <= N; ++j)
                for (int k = 0; k <= N; ++k) {
                    const double x = i*h, y = j*h, z = k*h;
                    const double f = std::sin(x)*std::cos(y)
                                   + std::sin(y)*std::cos(z)
                                   + std::sin(z)*std::cos(x);
                    acc.setValue(Coord(i, j, k), float(f));
                }
    }

    std::vector<Vec3s> points;
    std::vector<Vec3I> triangles;
    tools::marchingCubes(*grid, points, triangles, 0.0);

    ASSERT_FALSE(triangles.empty());

    const MeshStats s = analyze(points, triangles);
    EXPECT_EQ(size_t(0), s.boundaryEdges);     // watertight
    EXPECT_EQ(size_t(0), s.nonManifoldEdges);  // 2-manifold
    EXPECT_EQ(size_t(0), s.degenerateTris);
    // A gyroid is a high-genus surface (euler = 2 - 2*genus << 0); we only assert
    // watertight manifoldness, which is the MC33 guarantee.
    EXPECT_LT(s.euler, long(2));
}
