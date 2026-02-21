cmake_minimum_required(VERSION 3.24)
project(py_lap_solver)

# Find Python and pybind11
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Optional: OpenMP support
# On macOS with Homebrew, help CMake find libomp
if(APPLE)
    # Try to find libomp from Homebrew
    if(EXISTS "/opt/homebrew/opt/libomp")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    elseif(EXISTS "/usr/local/opt/libomp")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "/usr/local/opt/libomp/lib/libomp.dylib")
    endif()
endif()
find_package(OpenMP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include solver headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/batched_scipy)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/lap1015)

# Create the Python extension module for batched scipy solver
pybind11_add_module(_batched_scipy_lap
    src/cpp/bindings/batched_scipy_bindings.cpp
    src/cpp/batched_scipy/batched_scipy_lap.cpp
    src/cpp/batched_scipy/rectangular_lsap.cpp
)

# Create the Python extension module for LAP1015 solver
pybind11_add_module(_lap1015
    src/cpp/bindings/lap1015_bindings.cpp
)

# Compiler flags for optimization
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Base optimization flags
    set(BASE_FLAGS -O3 -fstrict-aliasing -flto)

    # Architecture-specific flags
    # When building wheels with cibuildwheel, builds run on native hardware for each arch
    # so -march=native is safe and optimal
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
        set(ARCH_FLAGS -mcpu=native)
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(ARCH_FLAGS -march=native -m64 -mfpmath=sse)
    endif()

    target_compile_options(_batched_scipy_lap PRIVATE ${BASE_FLAGS} ${ARCH_FLAGS})
    target_compile_options(_lap1015 PRIVATE ${BASE_FLAGS} ${ARCH_FLAGS})
    target_link_options(_batched_scipy_lap PRIVATE -flto)
    target_link_options(_lap1015 PRIVATE -flto)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(_batched_scipy_lap PRIVATE /O2 /GL)
    target_compile_options(_lap1015 PRIVATE /O2 /GL)
    target_link_options(_batched_scipy_lap PRIVATE /LTCG)
    target_link_options(_lap1015 PRIVATE /LTCG)
endif()

# Enable OpenMP if available
if (OpenMP_CXX_FOUND)
    target_link_libraries(_batched_scipy_lap PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_batched_scipy_lap PRIVATE LAP_OPENMP)
    message(STATUS "Batched scipy solver: OpenMP support enabled")

    target_link_libraries(_lap1015 PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_lap1015 PRIVATE LAP_OPENMP)
    message(STATUS "LAP1015 solver: OpenMP support enabled")
else()
    message(STATUS "Batched scipy solver: OpenMP not found - parallel support disabled")
    message(STATUS "LAP1015 solver: OpenMP not found - parallel support disabled")
endif()

# Define LAP solver options
target_compile_definitions(_lap1015 PRIVATE LAP_QUIET)  # Suppress debug output

# Install the extension modules
install(TARGETS _batched_scipy_lap _lap1015 LIBRARY DESTINATION py_lap_solver)

# ============================================================================
# CUDA support for LAP1015 Python bindings (optional)
# ============================================================================
include(CheckLanguage)

# Help CMake find nvcc inside Pixi/Conda environments where PATH probing can fail.
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    find_program(PY_LAP_NVCC nvcc
        HINTS
            "$ENV{CONDA_PREFIX}/bin"
            "$ENV{MAMBA_ROOT_PREFIX}/envs/default/bin"
            "$ENV{PIXI_PROJECT_ROOT}/.pixi/envs/default/bin"
    )
    if(PY_LAP_NVCC)
        set(CMAKE_CUDA_COMPILER "${PY_LAP_NVCC}" CACHE FILEPATH "CUDA compiler" FORCE)
        message(STATUS "LAP1015 CUDA solver: using nvcc at ${CMAKE_CUDA_COMPILER}")
    endif()
endif()

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    # Set architectures before enabling CUDA language; otherwise CMake's compiler
    # probe may inject legacy SM targets (e.g., compute_50) unsupported by CUDA 13+.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures" FORCE)
    endif()

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    pybind11_add_module(_lap1015_cuda
        src/cpp/bindings/lap1015_cuda_bindings.cu
    )

    target_compile_definitions(_lap1015_cuda PRIVATE LAP_CUDA LAP_QUIET)
    target_link_libraries(_lap1015_cuda PRIVATE CUDA::cudart)
    set_target_properties(_lap1015_cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
    )

    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(_lap1015_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3 -fstrict-aliasing ${ARCH_FLAGS}>
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda -O3>
        )
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(_lap1015_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/O2>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        )
    endif()

    if (OpenMP_CXX_FOUND)
        target_link_libraries(_lap1015_cuda PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(_lap1015_cuda PRIVATE LAP_OPENMP LAP_CUDA_OPENMP)
        message(STATUS "LAP1015 CUDA solver: OpenMP support enabled")
    else()
        message(STATUS "LAP1015 CUDA solver: OpenMP not found - multi-GPU parallel support disabled")
    endif()

    install(TARGETS _lap1015_cuda LIBRARY DESTINATION py_lap_solver)
    message(STATUS "LAP1015 CUDA solver: CUDA support enabled")
else()
    message(STATUS "LAP1015 CUDA solver: CUDA not found - GPU support disabled")
endif()
