import os

def open_file(file_path):
    """
        Opens a file.

        Args:
            file_path: Path of the file to open.

        Returns:
            The content of the file in an arbitrary format.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as file:
            return file.read()
    except UnicodeDecodeError:
        # If utf-8 decoding fails, try windows-1252.
        with open(file_path, "r", encoding="windows-1252", errors="replace") as file:
            return file.read()

def update_files():
    """
        Updates the content of files ending in .h, .cuh, .cc, .cu, and .cpp
        to call the appropriate API as we update NanoVDB from version 32.6 to
        version 32.7. This includes changes in namespaces, function names, and
        include directories.

        Args:
            None.

        Returns:
            None. Writes the contents of the file.
    """
    # Use current working directory to find files
    dir_path = os.getcwd()

    # List of file extensions to search for
    file_extensions = ['.h', '.cuh', '.cc', '.cu', '.cpp']

    nspace_dic = {
        'math': ['Ray', 'DDA<', 'HDDA', 'Vec3<', 'Vec4<', 'BBox<', 'ZeroCrossing', 'TreeMarcher', 'PointTreeMarcher', 'BoxStencil<', 'CurvatureStencil<', 'GradStencil<', 'WenoStencil<', 'AlignUp', 'Min', 'Max', 'Abs', 'Clamp', 'Sqrt', 'Sign', 'Maximum<', 'Delta<', 'RoundDown<', 'pi<', 'isApproxZero<', 'Round<', 'createSampler', 'SampleFromVoxels<'],
        'tools': ['createNanoGrid', 'StatsMode', 'createLevelSetSphere', 'createFogVolumeSphere', 'createFogVolumeSphere createFogVolumeSphere', 'createFogVolumeTorus', 'createLevelSetBox', 'CreateNanoGrid', 'updateGridStats', 'evalChecksum', 'validateChecksum', 'checkGrid', 'Extrema'],
        'util': ['is_floating_point', 'findLowestOn', 'findHighestOn', 'Range', 'streq', 'strcpy', 'strcat', 'empty(', 'Split', 'invoke', 'forEach', 'reduce', 'prefixSum', 'is_same', 'is_specialization', 'PtrAdd', 'PtrDiff'],
    }

    rename_dic = {
        # list from func4 in updateFiles.sh
        'nanovdb::build::': 'nanovdb::tools::build::',
        'nanovdb::BBoxR': 'nanovdb::Vec3dBBox',
        'nanovdb::BBox<nanovdb::Vec3d>': 'nanovdb::Vec3dBbox',
        # scope and rename, i.e. list from func2 in updateFiles.sh
        'nanovdb::cudaCreateNodeManager': 'nanovdb::cuda::createNodeManager',
        'nanovdb::cudaVoxelsToGrid': 'nanovdb::cuda::voxelsToGrid',
        'nanovdb::cudaPointsToGrid': 'nanovdb::cuda::pointsToGrid',
        'nanovdb::DitherLUT': 'nanovdb::math::DitherLUT',
        'nanovdb::PackedRGBA8': 'nanovdb::math::Rgba8',
        'nanovdb::Rgba8': 'nanovdb::math::Rgba8',
        'nanovdb::CpuTimer': 'nanovdb::util::Timer',
        'nanovdb::GpuTimer': 'nanovdb::util::cuda::Timer',
        'nanovdb::CountOn': 'nanovdb::util::countOn',
    }

    movdir_dic = {
        'util/GridHandle.h': 'GridHandle.h',
        'util/GridHandle.h': 'HostBuffer.h',
        'util/BuildGrid.h':   'tools/GridBuilder.h',
        'util/GridBuilder.h': 'tools/GridBuilder.h',
        'util/IO.h': 'io/IO.h',
        'util/CSampleFromVoxels.h': 'math/CSampleFromVoxels.h',
        'util/DitherLUT.h': 'math/DitherLUT.h',
        'util/HDDA.h': 'math/HDDA.h',
        'util/Ray.h': 'math/Ray.h',
        'util/SampleFromVoxels.h': 'math/SampleFromVoxels.h',
        'util/Stencils.h': 'nanovdb/math/Stencils.h',
        'util/CreateNanoGrid.h': 'tools/CreateNanoGrid.h',
        'util/Primitives.h': 'tools/CreatePrimitives.h',
        'util/GridChecksum.h': 'tools/GridChecksum.h',
        'util/GridStats.h': 'tools/GridStats.h',
        'util/GridChecksum.h': 'tools/GridChecksum.h',
        'util/GridValidator.h': 'tools/GridValidator.h',
        'util/NanoToOpenVDB.h': 'tools/NanoToOpenVDB.h',
        'util/cuda/CudaGridChecksum.cuh': 'tools/cuda/CudaGridChecksum.cuh',
        'util/cuda/CudaGridStats.cuh': 'tools/cuda/CudaGridStats.cuh',
        'util/cuda/CudaGridValidator.cuh': 'tools/cuda/CudaGridValidator.cuh',
        'util/cuda/CudaIndexToGrid.cuh': 'tools/cuda/CudaIndexToGrid.cuh',
        'util/cuda/CudaPointsToGrid.cuh': 'tools/GridChecksum.cuh',
        'util/cuda/CudaSignedFloodFill.cuh': 'tools/cuda/CudaSignedFloodFill.cuh',
        'util/cuda/CudaDeviceBuffer.h': 'cuda/DeviceBuffer.h',
        'util/cuda/CudaGridHandle.cuh': 'cuda/GridHandle.cuh',
        'util/cuda/CudaUtils.h': 'util/cuda/Util.h',
        'util/cuda/GpuTimer.h': 'util/cuda/Timer.h',
    }

    # Iterate over files in the directory and its subdirectories
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if any(file.endswith(ext) for ext in file_extensions):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")

                content = open_file(file_path)

                # Correspond to func1 $file in updateFiles.sh
                for key, vals in nspace_dic.items():
                    for val in vals:
                        new_word = key + '::' + val
                        content = content.replace(val, new_word)

                # Correspond to func4 and func2 in updateFiles.sh
                for key, val in rename_dic.items():
                    content = content.replace(key, val)

                # Correspond to func3 in updateFiles.sh
                for key, val in movdir_dic.items():
                    old_path = '<nanovdb/' + key + '>'
                    new_path = '<nanovdb/' + val + '>'
                    content = content.replace(old_path, new_path)

                with open(file_path, 'w') as file:
                    file.write(content)

if __name__ == "__main__":
    update_files()
