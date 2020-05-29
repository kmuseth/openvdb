///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) Ken Museth
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////

//#define BENCHMARK_FAST_SWEEPING
#define TIMING_FAST_SWEEPING

#include <sstream>
#include <cppunit/extensions/HelperMacros.h>
#include <openvdb/Types.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/ChangeBackground.h>
#include <openvdb/tools/Diagnostics.h>
#include <openvdb/tools/FastSweeping.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/LevelSetTracker.h>
#include <openvdb/tools/LevelSetRebuild.h>
#include <openvdb/tools/LevelSetPlatonic.h>
#include <openvdb/tools/LevelSetUtil.h>
#ifdef TIMING_FAST_SWEEPING
#include <openvdb/util/CpuTimer.h>
#endif

// Uncomment to test on models from our web-site
#define TestFastSweeping_DATA_PATH "/Users/ken/dev/data/vdb/"
//#define TestFastSweeping_DATA_PATH "/home/kmu/data/vdb/"
//#define TestFastSweeping_DATA_PATH "/usr/pic1/Data/OpenVDB/LevelSetModels/"

class TestFastSweeping: public CppUnit::TestFixture
{
public:
    virtual void setUp() { openvdb::initialize(); }
    virtual void tearDown() { openvdb::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestFastSweeping);
    CPPUNIT_TEST(dilateSignedDistance);
    CPPUNIT_TEST(testMaskSdf);
    CPPUNIT_TEST(testFogToSdf);
    CPPUNIT_TEST(testIntersection);
    CPPUNIT_TEST(velocityExtensionOfFogSphere);
    CPPUNIT_TEST(velocityExtensionOfSdfSphere);
    CPPUNIT_TEST(velocityExtensionOfFogBunny);
    CPPUNIT_TEST(velocityExtensionOfSdfBunny);
#ifdef BENCHMARK_FAST_SWEEPING
    CPPUNIT_TEST(testBenchmarks);
#endif

    CPPUNIT_TEST_SUITE_END();

    void dilateSignedDistance();
    void testMaskSdf();
    void testFogToSdf();
    void testIntersection();
    void velocityExtensionOfFogSphere();
    void velocityExtensionOfSdfSphere();
    void velocityExtensionOfFogBunny();
    void velocityExtensionOfSdfBunny();
#ifdef BENCHMARK_FAST_SWEEPING
    void testBenchmarks();
#endif
};// TestFastSweeping

CPPUNIT_TEST_SUITE_REGISTRATION(TestFastSweeping);

void
TestFastSweeping::dilateSignedDistance()
{
    using namespace openvdb;
    // Define parameters for the level set sphere to be re-normalized
    const float radius = 200.0f;
    const Vec3f center(0.0f, 0.0f, 0.0f);
    const float voxelSize = 1.0f;//half width
    const int width = 3, new_width = 50;//half width

    FloatGrid::Ptr grid = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, float(width));
    const size_t oldVoxelCount = grid->activeVoxelCount();

    tools::FastSweeping<FloatGrid> fs;
    CPPUNIT_ASSERT_EQUAL(size_t(0), fs.voxelCount());
    CPPUNIT_ASSERT_EQUAL(size_t(0), fs.boundaryCount());
    fs.initDilate(*grid, new_width - width);
    CPPUNIT_ASSERT(fs.voxelCount() > 0);
    CPPUNIT_ASSERT(fs.boundaryCount() > 0);
    fs.sweep();
    CPPUNIT_ASSERT(fs.voxelCount() > 0);
    CPPUNIT_ASSERT(fs.boundaryCount() > 0);
    auto grid2 = fs.sdfGrid();
    fs.clear();
    CPPUNIT_ASSERT_EQUAL(size_t(0), fs.voxelCount());
    CPPUNIT_ASSERT_EQUAL(size_t(0), fs.boundaryCount());
    const Index64 voxelCount = grid2->activeVoxelCount();
    CPPUNIT_ASSERT(voxelCount > oldVoxelCount);

    {// Check that the norm of the gradient for all active voxels is close to unity
        tools::Diagnose<FloatGrid> diagnose(*grid2);
        tools::CheckNormGrad<FloatGrid> test(*grid2, 0.99f, 1.01f);
        const std::string message = diagnose.check(test,
                                                   false,// don't generate a mask grid
                                                   true,// check active voxels
                                                   false,// ignore active tiles since a level set has none
                                                   false);// no need to check the background value
        CPPUNIT_ASSERT(message.empty());
        CPPUNIT_ASSERT_EQUAL(Index64(0), diagnose.failureCount());
        //std::cout << "\nOutput 1: " << message << std::endl;
    }
    {// Make sure all active voxels fail the following test
        tools::Diagnose<FloatGrid> diagnose(*grid2);
        tools::CheckNormGrad<FloatGrid> test(*grid2, std::numeric_limits<float>::min(), 0.99f);
        const std::string message = diagnose.check(test,
                                                   false,// don't generate a mask grid
                                                   true,// check active voxels
                                                   false,// ignore active tiles since a level set has none
                                                   false);// no need to check the background value
        CPPUNIT_ASSERT(!message.empty());
        CPPUNIT_ASSERT_EQUAL(voxelCount, diagnose.failureCount());
        //std::cout << "\nOutput 2: " << message << std::endl;
    }
    {// Make sure all active voxels fail the following test
        tools::Diagnose<FloatGrid> diagnose(*grid2);
        tools::CheckNormGrad<FloatGrid> test(*grid2, 1.01f, std::numeric_limits<float>::max());
        const std::string message = diagnose.check(test,
                                                   false,// don't generate a mask grid
                                                   true,// check active voxels
                                                   false,// ignore active tiles since a level set has none
                                                   false);// no need to check the background value
        CPPUNIT_ASSERT(!message.empty());
        CPPUNIT_ASSERT_EQUAL(voxelCount, diagnose.failureCount());
        //std::cout << "\nOutput 3: " << message << std::endl;
    }
}// dilateSignedDistance


void
TestFastSweeping::testMaskSdf()
{
    using namespace openvdb;
    // Define parameterS FOR the level set sphere to be re-normalized
    const float radius = 200.0f;
    const Vec3f center(0.0f, 0.0f, 0.0f);
    const float voxelSize = 1.0f, width = 3.0f;//half width
    const float new_width = 50;

    // Define a simple lambda function to write a grid to disk
    // This is useful for visual inspections in Houdini!
    auto writeFile = [](std::string name, FloatGrid::Ptr grid){
        io::File file(name);
        file.setCompression(io::COMPRESS_NONE);
        GridPtrVec grids;
        grids.push_back(grid);
        file.write(grids);
    };

    {// Use box as a mask
        //std::cerr << "\nUse box as a mask" << std::endl;
        FloatGrid::Ptr grid = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, width);
        CoordBBox bbox(Coord(150,-50,-50), Coord(250,50,50));
        MaskGrid mask;
        mask.sparseFill(bbox, true);

        writeFile("/tmp/box_mask_input.vdb", grid);
#ifdef TIMING_FAST_SWEEPING
        util::CpuTimer timer("\nParallel sparse fast sweeping with a box mask");
#endif
        grid = tools::maskSdf(*grid, mask);
        //tools::FastSweeping<FloatGrid> fs;
        //fs.initMask(*grid, mask);
        //fs.sweep();
        //std::cerr << "voxel count = " << fs.voxelCount() << std::endl;
        //std::cerr << "boundary count = " << fs.boundaryCount() << std::endl;
        //CPPUNIT_ASSERT(fs.voxelCount() > 0);
#ifdef TIMING_FAST_SWEEPING
        timer.stop();
#endif
        writeFile("/tmp/box_mask_output.vdb", grid);
        {// Check that the norm of the gradient for all active voxels is close to unity
            tools::Diagnose<FloatGrid> diagnose(*grid);
            tools::CheckNormGrad<FloatGrid> test(*grid, 0.99f, 1.01f);
            const std::string message = diagnose.check(test,
                                                       false,// don't generate a mask grid
                                                       true,// check active voxels
                                                       false,// ignore active tiles since a level set has none
                                                       false);// no need to check the background value
            //std::cerr << message << std::endl;
            const double percent = 100.0*double(diagnose.failureCount())/double(grid->activeVoxelCount());
            //std::cerr << "Failures = " << percent << "%" << std::endl;
            //std::cerr << "Failed: " << diagnose.failureCount() << std::endl;
            //std::cerr << "Total : " << grid->activeVoxelCount() << std::endl;
            CPPUNIT_ASSERT(percent < 0.01);
            //CPPUNIT_ASSERT(message.empty());
            //CPPUNIT_ASSERT_EQUAL(size_t(0), diagnose.failureCount());
        }
    }

    {// Use sphere as a mask
        //std::cerr << "\nUse sphere as a mask" << std::endl;
        FloatGrid::Ptr grid = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, width);
        FloatGrid::Ptr mask = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, new_width);

        writeFile("/tmp/sphere_mask_input.vdb", grid);
#ifdef TIMING_FAST_SWEEPING
        util::CpuTimer timer("\nParallel sparse fast sweeping with a sphere mask");
#endif
        grid = tools::maskSdf(*grid, *mask);
        //tools::FastSweeping<FloatGrid> fs;
        //fs.initMask(*grid, *mask);
        //fs.sweep();
#ifdef TIMING_FAST_SWEEPING
        timer.stop();
#endif
        //std::cerr << "voxel count = " << fs.voxelCount() << std::endl;
        //std::cerr << "boundary count = " << fs.boundaryCount() << std::endl;
        //CPPUNIT_ASSERT(fs.voxelCount() > 0);
        writeFile("/tmp/sphere_mask_output.vdb", grid);
        {// Check that the norm of the gradient for all active voxels is close to unity
            tools::Diagnose<FloatGrid> diagnose(*grid);
            tools::CheckNormGrad<FloatGrid> test(*grid, 0.99f, 1.01f);
            const std::string message = diagnose.check(test,
                                                       false,// don't generate a mask grid
                                                       true,// check active voxels
                                                       false,// ignore active tiles since a level set has none
                                                       false);// no need to check the background value
            //std::cerr << message << std::endl;
            const double percent = 100.0*double(diagnose.failureCount())/double(grid->activeVoxelCount());
            //std::cerr << "Failures = " << percent << "%" << std::endl;
            //std::cerr << "Failed: " << diagnose.failureCount() << std::endl;
            //std::cerr << "Total : " << grid->activeVoxelCount() << std::endl;
            //CPPUNIT_ASSERT(message.empty());
            //CPPUNIT_ASSERT_EQUAL(size_t(0), diagnose.failureCount());
            CPPUNIT_ASSERT(percent < 0.01);
            //std::cout << "\nOutput 1: " << message << std::endl;
        }
    }

    {// Use dodecahedron as a mask
        //std::cerr << "\nUse dodecahedron as a mask" << std::endl;
        FloatGrid::Ptr grid = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, width);
        FloatGrid::Ptr mask = tools::createLevelSetDodecahedron<FloatGrid>(50, Vec3f(radius, 0.0f, 0.0f),
                                                                           voxelSize, 10);

        writeFile("/tmp/dodecahedron_mask_input.vdb", grid);
#ifdef TIMING_FAST_SWEEPING
        util::CpuTimer timer("\nParallel sparse fast sweeping with a dodecahedron mask");
#endif
        grid = tools::maskSdf(*grid, *mask);
        //tools::FastSweeping<FloatGrid> fs;
        //fs.initMask(*grid, *mask);
        //std::cerr << "voxel count = " << fs.voxelCount() << std::endl;
        //std::cerr << "boundary count = " << fs.boundaryCount() << std::endl;
        //CPPUNIT_ASSERT(fs.voxelCount() > 0);
        //fs.sweep();
#ifdef TIMING_FAST_SWEEPING
        timer.stop();
#endif
        writeFile("/tmp/dodecahedron_mask_output.vdb", grid);
        {// Check that the norm of the gradient for all active voxels is close to unity
            tools::Diagnose<FloatGrid> diagnose(*grid);
            tools::CheckNormGrad<FloatGrid> test(*grid, 0.99f, 1.01f);
            const std::string message = diagnose.check(test,
                                                       false,// don't generate a mask grid
                                                       true,// check active voxels
                                                       false,// ignore active tiles since a level set has none
                                                       false);// no need to check the background value
            //std::cerr << message << std::endl;
            const double percent = 100.0*double(diagnose.failureCount())/double(grid->activeVoxelCount());
            //std::cerr << "Failures = " << percent << "%" << std::endl;
            //std::cerr << "Failed: " << diagnose.failureCount() << std::endl;
            //std::cerr << "Total : " << grid->activeVoxelCount() << std::endl;
            //CPPUNIT_ASSERT(message.empty());
            //CPPUNIT_ASSERT_EQUAL(size_t(0), diagnose.failureCount());
            CPPUNIT_ASSERT(percent < 0.01);
            //std::cout << "\nOutput 1: " << message << std::endl;
        }
    }
#ifdef TestFastSweeping_DATA_PATH
     {// Use bunny as a mask
         //std::cerr << "\nUse bunny as a mask" << std::endl;
         FloatGrid::Ptr grid = tools::createLevelSetSphere<FloatGrid>(10.0f, Vec3f(-10,0,0), 0.05f, width);
         openvdb::initialize();//required whenever I/O of OpenVDB files is performed!
         const std::string path(TestFastSweeping_DATA_PATH);
         io::File file( path + "bunny.vdb" );
         file.open(false);//disable delayed loading
         FloatGrid::Ptr mask = openvdb::gridPtrCast<openvdb::FloatGrid>(file.getGrids()->at(0));

         writeFile("/tmp/bunny_mask_input.vdb", grid);
         tools::FastSweeping<FloatGrid> fs;
#ifdef TIMING_FAST_SWEEPING
         util::CpuTimer timer("\nParallel sparse fast sweeping with a bunny mask");
#endif
         fs.initMask(*grid, *mask);
         //std::cerr << "voxel count = " << fs.voxelCount() << std::endl;
         //std::cerr << "boundary count = " << fs.boundaryCount() << std::endl;
         fs.sweep();
         auto grid2 = fs.sdfGrid();
#ifdef TIMING_FAST_SWEEPING
         timer.stop();
#endif
         writeFile("/tmp/bunny_mask_output.vdb", grid2);
         {// Check that the norm of the gradient for all active voxels is close to unity
             tools::Diagnose<FloatGrid> diagnose(*grid2);
             tools::CheckNormGrad<FloatGrid> test(*grid2, 0.99f, 1.01f);
             const std::string message = diagnose.check(test,
                                                        false,// don't generate a mask grid
                                                        true,// check active voxels
                                                        false,// ignore active tiles since a level set has none
                                                        false);// no need to check the background value
             //std::cerr << message << std::endl;
             const double percent = 100.0*double(diagnose.failureCount())/double(grid2->activeVoxelCount());
             //std::cerr << "Failures = " << percent << "%" << std::endl;
             //std::cerr << "Failed: " << diagnose.failureCount() << std::endl;
             //std::cerr << "Total : " << grid2->activeVoxelCount() << std::endl;
             //CPPUNIT_ASSERT(message.empty());
             //CPPUNIT_ASSERT_EQUAL(size_t(0), diagnose.failureCount());
             CPPUNIT_ASSERT(percent < 4.5);// crossing characteristics!
             //std::cout << "\nOutput 1: " << message << std::endl;
         }
     }
#endif
}// testMaskSdf

void
TestFastSweeping::testFogToSdf()
{
    using namespace openvdb;
    // Define parameterS FOR the level set sphere to be re-normalized
    const float radius = 200.0f;
    const Vec3f center(0.0f, 0.0f, 0.0f);
    const float voxelSize = 1.0f, width = 3.0f;//half width

    FloatGrid::Ptr grid = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, float(width));
    tools::sdfToFogVolume(*grid);
    const Index64 voxelCount = grid->activeVoxelCount();

    // Define a simple lambda function to write a grid to disk
    // This is useful for visual inspections in Houdini!
    auto writeFile = [](std::string name, FloatGrid::Ptr grid){
        io::File file(name);
        file.setCompression(io::COMPRESS_NONE);
        GridPtrVec grids;
        grids.push_back(grid);
        file.write(grids);
    };

    writeFile("/tmp/fog_input.vdb", grid);
    tools::FastSweeping<FloatGrid> fs;
#ifdef TIMING_FAST_SWEEPING
    util::CpuTimer timer("\nParallel sparse fast sweeping with a fog volume");
#endif
    fs.initSdf(*grid, /*isoValue*/0.5f,/*isInputSdf*/false);
    CPPUNIT_ASSERT(fs.voxelCount() > 0);
    std::cerr << "voxel count = " << fs.voxelCount() << std::endl;
    std::cerr << "boundary count = " << fs.boundaryCount() << std::endl;
    fs.sweep();
    auto grid2 = fs.sdfGrid();
#ifdef TIMING_FAST_SWEEPING
    timer.stop();
#endif
    CPPUNIT_ASSERT_EQUAL(voxelCount, grid->activeVoxelCount());
    writeFile("/tmp/ls_output.vdb", grid2);

    {// Check that the norm of the gradient for all active voxels is close to unity
        tools::Diagnose<FloatGrid> diagnose(*grid2);
        tools::CheckNormGrad<FloatGrid> test(*grid2, 0.99f, 1.01f);
        const std::string message = diagnose.check(test,
                                                   false,// don't generate a mask grid
                                                   true,// check active voxels
                                                   false,// ignore active tiles since a level set has none
                                                   false);// no need to check the background value
        std::cerr << message << std::endl;
        const double percent = 100.0*double(diagnose.failureCount())/double(grid2->activeVoxelCount());
        std::cerr << "Failures = " << percent << "%" << std::endl;
        std::cerr << "Failure count = " << diagnose.failureCount() << std::endl;
        std::cerr << "Total active voxel count = " << grid2->activeVoxelCount() << std::endl;
        CPPUNIT_ASSERT(percent < 3.0);
    }
}// testFogToSdf


#ifdef BENCHMARK_FAST_SWEEPING
void
TestFastSweeping::testBenchmarks()
{
    using namespace openvdb;
    // Define parameterS FOR the level set sphere to be re-normalized
    const float radius = 200.0f;
    const Vec3f center(0.0f, 0.0f, 0.0f);
    const float voxelSize = 1.0f, width = 3.0f;//half width
    const float new_width = 50;

    // Define a simple lambda function to write a grid to disk
    // This is useful for visual inspections in Houdini!
    auto writeFile = [](std::string name, FloatGrid::Ptr grid){
        io::File file(name);
        file.setCompression(io::COMPRESS_NONE);
        GridPtrVec grids;
        grids.push_back(grid);
        file.write(grids);
    };

    {// Use rebuildLevelSet (limited to closed and symmetric narrow-band level sets)
        FloatGrid::Ptr grid = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, width);
#ifdef TIMING_FAST_SWEEPING
        util::CpuTimer timer("\nRebuild level set");
#endif
        FloatGrid::Ptr ls = tools::levelSetRebuild(*grid, 0.0f, new_width);
#ifdef TIMING_FAST_SWEEPING
        timer.stop();
#endif
        std::cout << "Diagnostics:\n" << tools::checkLevelSet(*ls, 9) << std::endl;
        writeFile("/tmp/rebuild_sdf.vdb", ls);
    }
    {// Use LevelSetTracker::normalize()
        FloatGrid::Ptr grid = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, width);
        tools::dilateActiveValues(grid->tree(), int(new_width-width), tools::NN_FACE, tools::IGNORE_TILES);
        tools::changeLevelSetBackground(grid->tree(), new_width);
        std::cout << "Diagnostics:\n" << tools::checkLevelSet(*grid, 9) << std::endl;
        //std::cerr << "Number of active tiles = " << grid->tree().activeTileCount() << std::endl;
        //grid->print(std::cout, 3);
        tools::LevelSetTracker<FloatGrid> track(*grid);
        track.setNormCount(int(new_width/0.3f));//CFL is 1/3 for RK1
        track.setSpatialScheme(math::FIRST_BIAS);
        track.setTemporalScheme(math::TVD_RK1);
#ifdef TIMING_FAST_SWEEPING
        util::CpuTimer timer("\nConventional re-normalization");
#endif
        track.normalize();
#ifdef TIMING_FAST_SWEEPING
        timer.stop();
#endif
        std::cout << "Diagnostics:\n" << tools::checkLevelSet(*grid, 9) << std::endl;
        writeFile("/tmp/old_sdf.vdb", grid);
    }
    {// Use new sparse and parallel fast sweeping
        FloatGrid::Ptr grid = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, width);

        writeFile("/tmp/original_sdf.vdb", grid);
#ifdef TIMING_FAST_SWEEPING
        util::CpuTimer timer("\nParallel sparse fast sweeping");
#endif
        auto grid2 = tools::dilateSdf(*grid, int(new_width - width), tools::NN_FACE_EDGE);
        //tools::FastSweeping<FloatGrid> fs(*grid);
        //CPPUNIT_ASSERT(fs.voxelCount() > 0);
        //tbb::task_scheduler_init init(4);//thread count
        //fs.sweep();
#ifdef TIMING_FAST_SWEEPING
        timer.stop();
#endif
        //std::cout << "Diagnostics:\n" << tools::checkLevelSet(*grid, 9) << std::endl;
        writeFile("/tmp/new_sdf.vdb", grid2);
    }
}
#endif

void
TestFastSweeping::testIntersection()
{
  using namespace openvdb;
  const Coord ijk(1,4,-9);
  FloatGrid grid(0.0f);
  auto acc = grid.getAccessor();
  math::GradStencil<FloatGrid> stencil(grid);
  acc.setValue(ijk,-1.0f);
  int cases = 0;
  for (int mx=0; mx<2; ++mx) {
    acc.setValue(ijk.offsetBy(-1,0,0), mx ? 1.0f : -1.0f);
    for (int px=0; px<2; ++px) {
      acc.setValue(ijk.offsetBy(1,0,0), px ? 1.0f : -1.0f);
      for (int my=0; my<2; ++my) {
        acc.setValue(ijk.offsetBy(0,-1,0), my ? 1.0f : -1.0f);
        for (int py=0; py<2; ++py) {
          acc.setValue(ijk.offsetBy(0,1,0), py ? 1.0f : -1.0f);
          for (int mz=0; mz<2; ++mz) {
            acc.setValue(ijk.offsetBy(0,0,-1), mz ? 1.0f : -1.0f);
            for (int pz=0; pz<2; ++pz) {
              acc.setValue(ijk.offsetBy(0,0,1), pz ? 1.0f : -1.0f);
              ++cases;
              CPPUNIT_ASSERT_EQUAL(Index64(7), grid.activeVoxelCount());
              stencil.moveTo(ijk);
              const size_t count = mx + px + my + py + mz + pz;// number of intersections
              CPPUNIT_ASSERT(stencil.intersects() == (count > 0));
              auto mask = stencil.intersectionMask();
              CPPUNIT_ASSERT(mask.none() == (count == 0));
              CPPUNIT_ASSERT(mask.any() == (count > 0));
              CPPUNIT_ASSERT_EQUAL(count, mask.count());
              CPPUNIT_ASSERT(mask.test(0) == mx);
              CPPUNIT_ASSERT(mask.test(1) == px);
              CPPUNIT_ASSERT(mask.test(2) == my);
              CPPUNIT_ASSERT(mask.test(3) == py);
              CPPUNIT_ASSERT(mask.test(4) == mz);
              CPPUNIT_ASSERT(mask.test(5) == pz);
            }//pz
          }//mz
        }//py
      }//my
    }//px
  }//mx
  CPPUNIT_ASSERT_EQUAL(64, cases);// = 2^6
}//testIntersection

void
TestFastSweeping::velocityExtensionOfFogSphere()
{
  using namespace openvdb;
  auto writeFile = [](std::string name, FloatGrid::Ptr grid){
    io::File file(name);
    file.setCompression(io::COMPRESS_NONE);
    GridPtrVec grids;
    grids.push_back(grid);
    file.write(grids);
  };
  const float isoValue = 0.5f;
  const float radius = 100.0f;
  const Vec3f center(0.0f, 0.0f, 0.0f);
  const float voxelSize = 1.0f, width = 3.0f;//half width
  const float inside = -std::numeric_limits<float>::min();
  const float outside = std::numeric_limits<float>::max();
  FloatGrid::Ptr grid = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, float(width));
  tools::sdfToFogVolume(*grid);
  writeFile("/tmp/sphere1_fog_in.vdb", grid);

  //tools::fogToSdf(*grid, isoValue);
  auto op = [radius](const Vec3R &xyz) {
    return math::Sin(2*3.14*xyz[0]/radius);
    //return xyz[0]>0 ? 0.5f : -0.5f;
  };
  auto grids = tools::fogToSdfAndExt(*grid, op, isoValue);
  writeFile("/tmp/sphere1_sdf_out.vdb", grids[0]);
  writeFile("/tmp/sphere1_ext_out.vdb", grids[1]);
}//velocityExtensionOfFogSphere

void
TestFastSweeping::velocityExtensionOfSdfSphere()
{
  using namespace openvdb;
  auto writeFile = [](std::string name, FloatGrid::Ptr grid){
    io::File file(name);
    file.setCompression(io::COMPRESS_NONE);
    GridPtrVec grids;
    grids.push_back(grid);
    file.write(grids);
  };
  const float isoValue = 0.0f;
  const float radius = 100.0f;
  const Vec3f center(0.0f, 0.0f, 0.0f);
  const float voxelSize = 1.0f, width = 10.0f;//half width
  const float inside = -std::numeric_limits<float>::min();
  const float outside = std::numeric_limits<float>::max();
  FloatGrid::Ptr grid = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, float(width));
  writeFile("/tmp/sphere2_sdf_in.vdb", grid);

  auto op = [radius](const Vec3R &xyz) {
    return math::Sin(2*3.14*xyz[0]/radius);
    //return xyz[0]>0 ? 0.5f : -0.5f;
  };
  auto grids = tools::sdfToSdfAndExt(*grid, op, isoValue);
  writeFile("/tmp/sphere2_sdf_out.vdb", grids[0]);
  writeFile("/tmp/sphere2_ext_out.vdb", grids[1]);
}//velocityExtensionOfSdfSphere

void
TestFastSweeping::velocityExtensionOfFogBunny()
{
  using namespace openvdb;
  auto writeFile = [](std::string name, FloatGrid::Ptr grid){
    io::File file(name);
    file.setCompression(io::COMPRESS_NONE);
    GridPtrVec grids;
    grids.push_back(grid);
    file.write(grids);
  };

  openvdb::initialize();//required whenever I/O of OpenVDB files is performed!
  const std::string path(TestFastSweeping_DATA_PATH);
  io::File file( path + "bunny.vdb" );
  file.open(false);//disable delayed loading
  auto grid = openvdb::gridPtrCast<openvdb::FloatGrid>(file.getGrids()->at(0));
  tools::sdfToFogVolume(*grid);
  writeFile("/tmp/bunny1_fog_in.vdb", grid);
  auto bbox = grid->evalActiveVoxelBoundingBox();
  const double xSize = bbox.dim()[0]*grid->voxelSize()[0];
  std::cerr << "\ndim=" << bbox.dim() << ", voxelSize="<< grid->voxelSize()[0]
            << ", xSize=" << xSize << std::endl;

  auto op = [xSize](const Vec3R &xyz) {
    return math::Sin(2*3.14*xyz[0]/xSize);
  };
  auto grids = tools::fogToSdfAndExt(*grid, op, 0.5f);
  std::cerr << "before writing" << std::endl;
  writeFile("/tmp/bunny1_sdf_out.vdb", grids[0]);
  writeFile("/tmp/bunny1_ext_out.vdb", grids[1]);
  std::cerr << "after writing" << std::endl;
}//velocityExtensionOfFogBunnyevalActiveVoxelBoundingBox

void
TestFastSweeping::velocityExtensionOfSdfBunny()
{
  using namespace openvdb;
  auto writeFile = [](std::string name, FloatGrid::Ptr grid){
    io::File file(name);
    file.setCompression(io::COMPRESS_NONE);
    GridPtrVec grids;
    grids.push_back(grid);
    file.write(grids);
  };

  openvdb::initialize();//required whenever I/O of OpenVDB files is performed!
  const std::string path(TestFastSweeping_DATA_PATH);
  io::File file( path + "bunny.vdb" );
  file.open(false);//disable delayed loading
  auto grid = openvdb::gridPtrCast<openvdb::FloatGrid>(file.getGrids()->at(0));
  writeFile("/tmp/bunny2_sdf_in.vdb", grid);
  auto bbox = grid->evalActiveVoxelBoundingBox();
  const double xSize = bbox.dim()[0]*grid->voxelSize()[0];
  std::cerr << "\ndim=" << bbox.dim() << ", voxelSize="<< grid->voxelSize()[0]
            << ", xSize=" << xSize << std::endl;

  auto op = [xSize](const Vec3R &xyz) {
    return math::Sin(2*3.14*xyz[0]/xSize);
  };
  auto grids = tools::sdfToSdfAndExt(*grid, op);
  std::cerr << "before writing" << std::endl;
  writeFile("/tmp/bunny2_sdf_out.vdb", grids[0]);
  writeFile("/tmp/bunny2_ext_out.vdb", grids[1]);
  std::cerr << "after writing" << std::endl;
}//velocityExtensionOfFogBunnyevalActiveVoxelBoundingBox

// Copyright (c) Ken Museth
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )