#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <memory>
#include <vector>

#include "lap.h"

namespace py = pybind11;

#ifdef LAP_CUDA
struct SolveTiming {
    double transfer_ms;
    double solve_ms;
    double total_ms;
};

template <class SC, class TC, class CF>
void solveTableCUDA(
    int N1,
    int N2,
    CF &get_cost,
    int *rowsol,
    bool use_epsilon,
    bool use_pinned_memory,
    int max_devices,
    std::vector<int> devices,
    bool silent,
    bool sequential_cost_init = false)
{
    lap::cuda::CpuCostFunction<TC, CF> costFunction(get_cost, sequential_cost_init);
    lap::cuda::Worksharing ws(N2, 8, devices, max_devices, silent);
    // Build CPU table once, then upload full per-device partitions to GPU once.
    // This avoids repeated host->device row transfers during the solve.
    std::unique_ptr<lap::cuda::GPUTableCost<TC>> gpuCostMatrix;
    {
        lap::cuda::CPUTableCost<TC> cpuCostMatrix(N1, N2, costFunction, ws, use_pinned_memory);
        gpuCostMatrix = std::make_unique<lap::cuda::GPUTableCost<TC>>(N1, N2, cpuCostMatrix, ws);
    }

    lap::cuda::DirectIterator<TC, lap::cuda::GPUTableCost<TC>> iterator(*gpuCostMatrix, ws);
    lap::cuda::solve<SC, TC>(N1, N2, *gpuCostMatrix, iterator, rowsol, use_epsilon);
}

template <class SC, class TC, class CF>
SolveTiming solveTableCUDAWithTiming(
    int N1,
    int N2,
    CF &get_cost,
    int *rowsol,
    bool use_epsilon,
    bool use_pinned_memory,
    int max_devices,
    std::vector<int> devices,
    bool silent,
    bool sequential_cost_init = false)
{
    using Clock = std::chrono::steady_clock;
    const auto total_start = Clock::now();

    lap::cuda::CpuCostFunction<TC, CF> costFunction(get_cost, sequential_cost_init);
    lap::cuda::Worksharing ws(N2, 8, devices, max_devices, silent);

    std::unique_ptr<lap::cuda::GPUTableCost<TC>> gpuCostMatrix;
    std::chrono::time_point<Clock> transfer_start;
    std::chrono::time_point<Clock> transfer_end;

    {
        lap::cuda::CPUTableCost<TC> cpuCostMatrix(N1, N2, costFunction, ws, use_pinned_memory);
        // Measure only the host-table -> device upload phase.
        transfer_start = Clock::now();
        gpuCostMatrix = std::make_unique<lap::cuda::GPUTableCost<TC>>(N1, N2, cpuCostMatrix, ws);
        transfer_end = Clock::now();
    }

    lap::cuda::DirectIterator<TC, lap::cuda::GPUTableCost<TC>> iterator(*gpuCostMatrix, ws);
    const auto solve_start = Clock::now();
    lap::cuda::solve<SC, TC>(N1, N2, *gpuCostMatrix, iterator, rowsol, use_epsilon);
    const auto solve_end = Clock::now();

    const auto transfer_ms = std::chrono::duration<double, std::milli>(transfer_end - transfer_start).count();
    const auto solve_ms = std::chrono::duration<double, std::milli>(solve_end - solve_start).count();
    const auto total_ms = std::chrono::duration<double, std::milli>(solve_end - total_start).count();

    return SolveTiming{transfer_ms, solve_ms, total_ms};
}
#endif

py::array_t<int32_t> solve_lap_cuda_float(
    py::array_t<float> cost_matrix,
    int num_valid = -1,
    bool use_epsilon = true,
    bool use_pinned_memory = true,
    int max_devices = 1,
    std::vector<int> devices = {},
    bool silent = true)
{
#ifndef LAP_CUDA
    throw std::runtime_error("CUDA support not enabled");
#else
    auto C = cost_matrix.unchecked<2>();
    if (C.ndim() != 2) {
        throw std::runtime_error("The cost matrix must be 2-dimensional");
    }

    const int Nx = C.shape(0);
    const int Ny = C.shape(1);
    const int dim_rows = (num_valid > 0) ? num_valid : Nx;
    const int dim_cols = Ny;

    if (dim_rows > Nx) {
        throw std::runtime_error("num_valid cannot exceed number of rows in cost_matrix");
    }
    if (dim_rows > dim_cols) {
        throw std::runtime_error("LAP1015 CUDA expects rows <= cols; transpose before calling");
    }
    if (max_devices <= 0) {
        throw std::runtime_error("max_devices must be >= 1");
    }

    auto get_cost = [&C](int x, int y) -> float { return C(x, y); };
    std::unique_ptr<int[]> rowsol(new int[dim_cols]);

    solveTableCUDA<float, float>(
        dim_rows,
        dim_cols,
        get_cost,
        rowsol.get(),
        use_epsilon,
        use_pinned_memory,
        max_devices,
        devices,
        silent);

    py::array_t<int32_t, py::array::c_style> result(Nx);
    auto r = result.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < Nx; ++i) {
        r(i) = (i < dim_rows) ? rowsol[i] : -1;
    }
    return result;
#endif
}

py::tuple solve_lap_cuda_float_timed(
    py::array_t<float> cost_matrix,
    int num_valid = -1,
    bool use_epsilon = true,
    bool use_pinned_memory = true,
    int max_devices = 1,
    std::vector<int> devices = {},
    bool silent = true)
{
#ifndef LAP_CUDA
    throw std::runtime_error("CUDA support not enabled");
#else
    auto C = cost_matrix.unchecked<2>();
    if (C.ndim() != 2) {
        throw std::runtime_error("The cost matrix must be 2-dimensional");
    }

    const int Nx = C.shape(0);
    const int Ny = C.shape(1);
    const int dim_rows = (num_valid > 0) ? num_valid : Nx;
    const int dim_cols = Ny;

    if (dim_rows > Nx) {
        throw std::runtime_error("num_valid cannot exceed number of rows in cost_matrix");
    }
    if (dim_rows > dim_cols) {
        throw std::runtime_error("LAP1015 CUDA expects rows <= cols; transpose before calling");
    }
    if (max_devices <= 0) {
        throw std::runtime_error("max_devices must be >= 1");
    }

    auto get_cost = [&C](int x, int y) -> float { return C(x, y); };
    std::unique_ptr<int[]> rowsol(new int[dim_cols]);

    const SolveTiming timing = solveTableCUDAWithTiming<float, float>(
        dim_rows,
        dim_cols,
        get_cost,
        rowsol.get(),
        use_epsilon,
        use_pinned_memory,
        max_devices,
        devices,
        silent);

    py::array_t<int32_t, py::array::c_style> result(Nx);
    auto r = result.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < Nx; ++i) {
        r(i) = (i < dim_rows) ? rowsol[i] : -1;
    }

    py::dict timing_dict;
    timing_dict["transfer_ms"] = timing.transfer_ms;
    timing_dict["solve_ms"] = timing.solve_ms;
    timing_dict["total_ms"] = timing.total_ms;

    return py::make_tuple(result, timing_dict);
#endif
}

py::array_t<int32_t> solve_lap_cuda_double(
    py::array_t<double> cost_matrix,
    int num_valid = -1,
    bool use_epsilon = true,
    bool use_pinned_memory = true,
    int max_devices = 1,
    std::vector<int> devices = {},
    bool silent = true)
{
#ifndef LAP_CUDA
    throw std::runtime_error("CUDA support not enabled");
#else
    auto C = cost_matrix.unchecked<2>();
    if (C.ndim() != 2) {
        throw std::runtime_error("The cost matrix must be 2-dimensional");
    }

    const int Nx = C.shape(0);
    const int Ny = C.shape(1);
    const int dim_rows = (num_valid > 0) ? num_valid : Nx;
    const int dim_cols = Ny;

    if (dim_rows > Nx) {
        throw std::runtime_error("num_valid cannot exceed number of rows in cost_matrix");
    }
    if (dim_rows > dim_cols) {
        throw std::runtime_error("LAP1015 CUDA expects rows <= cols; transpose before calling");
    }
    if (max_devices <= 0) {
        throw std::runtime_error("max_devices must be >= 1");
    }

    auto get_cost = [&C](int x, int y) -> double { return C(x, y); };
    std::unique_ptr<int[]> rowsol(new int[dim_cols]);

    solveTableCUDA<double, double>(
        dim_rows,
        dim_cols,
        get_cost,
        rowsol.get(),
        use_epsilon,
        use_pinned_memory,
        max_devices,
        devices,
        silent);

    py::array_t<int32_t, py::array::c_style> result(Nx);
    auto r = result.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < Nx; ++i) {
        r(i) = (i < dim_rows) ? rowsol[i] : -1;
    }
    return result;
#endif
}

int get_cuda_device_count()
{
#ifdef LAP_CUDA
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return count;
#else
    return 0;
#endif
}

PYBIND11_MODULE(_lap1015_cuda, m)
{
    m.doc() = "LAP1015 CUDA solver bindings";

    m.def(
        "solve_lap_cuda_float",
        &solve_lap_cuda_float,
        py::arg("cost_matrix"),
        py::arg("num_valid") = -1,
        py::arg("use_epsilon") = true,
        py::arg("use_pinned_memory") = true,
        py::arg("max_devices") = 1,
        py::arg("devices") = std::vector<int>{},
        py::arg("silent") = true,
        "Solve LAP on CUDA with single precision (float32)");

    m.def(
        "solve_lap_cuda_double",
        &solve_lap_cuda_double,
        py::arg("cost_matrix"),
        py::arg("num_valid") = -1,
        py::arg("use_epsilon") = true,
        py::arg("use_pinned_memory") = true,
        py::arg("max_devices") = 1,
        py::arg("devices") = std::vector<int>{},
        py::arg("silent") = true,
        "Solve LAP on CUDA with double precision (float64)");

    m.def(
        "solve_lap_cuda_float_timed",
        &solve_lap_cuda_float_timed,
        py::arg("cost_matrix"),
        py::arg("num_valid") = -1,
        py::arg("use_epsilon") = true,
        py::arg("use_pinned_memory") = true,
        py::arg("max_devices") = 1,
        py::arg("devices") = std::vector<int>{},
        py::arg("silent") = true,
        "Solve LAP on CUDA with float32 and return (assignment, timing dict).");

    m.def("get_cuda_device_count", &get_cuda_device_count, "Get number of visible CUDA devices");

#ifdef LAP_CUDA
    m.attr("HAS_CUDA") = true;
#else
    m.attr("HAS_CUDA") = false;
#endif
}
