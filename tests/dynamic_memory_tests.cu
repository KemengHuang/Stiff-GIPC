#include <cuda_tools/cuda_all.h>
#include <gipc/utils/parallel_algorithm/fast_segmental_reduce.h>
#include <linear_system/linear_system/global_matrix.h>
#include <solver/MASPreconditioner.cuh>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

namespace
{
static_assert(!std::is_copy_constructible_v<cudatool::DeviceBuffer<int>>);
static_assert(!std::is_copy_assignable_v<cudatool::DeviceBuffer<int>>);
static_assert(std::is_move_constructible_v<cudatool::DeviceBuffer<int>>);

void require(bool condition, const char* message)
{
    if(!condition)
    {
        std::fprintf(stderr, "dynamic-memory test failed: %s\n", message);
        std::abort();
    }
}

__global__ void fill_sequence(int* values, int n, int base)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        values[i] = base + i;
}

__global__ void write_scalar(int* value, int input)
{
    if(blockIdx.x == 0 && threadIdx.x == 0)
        *value = input;
}

__global__ void synthetic_detect(int       candidate_count,
                                 uint32_t  capacity,
                                 int4*     pairs,
                                 int4*     ccd_pairs,
                                 int*      matrix_indices,
                                 uint32_t* count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= candidate_count)
        return;

    uint32_t slot = atomicAdd(count, 1U);
    if(slot >= capacity)
        return;

    pairs[slot]          = make_int4(i, i + 1, i + 2, i + 3);
    ccd_pairs[slot]      = make_int4(-i - 1, i + 1, i + 2, i + 3);
    matrix_indices[slot] = i * 3;
}

__global__ void synthetic_ccd_detect(int candidate_count, uint32_t capacity, int4* ccd_pairs, uint32_t* count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= candidate_count)
        return;

    uint32_t slot = atomicAdd(count, 1U);
    if(slot < capacity)
        ccd_pairs[slot] = make_int4(i, i + 1, i + 2, i + 3);
}

void test_device_buffer_preserve()
{
    cudatool::DeviceBuffer<int> values;
    require(values.size() == 0 && values.capacity() == 0,
            "DeviceBuffer must start with zero size and capacity");
    auto empty_view = values.view(17);
    require(empty_view.size() == 0 && empty_view.data() == nullptr,
            "out-of-range view of an empty buffer must stay empty and null");

    values.resize_discard(4);
    fill_sequence<<<1, 32>>>(values.data(), 4, 10);
    checkCudaErrors(cudaDeviceSynchronize());

    values.resize(13);
    std::vector<int> host;
    values.copy_to(host);
    require(host.size() == 13, "resize must update logical size");
    for(int i = 0; i < 4; ++i)
        require(host[i] == 10 + i, "vector-like resize growth lost data");

    fill_sequence<<<1, 32>>>(values.data(), 13, 20);
    checkCudaErrors(cudaDeviceSynchronize());
    values.resize_preserve(37);
    values.copy_to(host);
    for(int i = 0; i < 13; ++i)
        require(host[i] == 20 + i, "second preserve growth lost extended data");
}

void test_launcher_does_not_copy_owning_buffers()
{
    cudatool::DeviceBuffer<int> value(1);
    value.fill(0);
    int* original_allocation = value.data();

    LaunchCudaKernal_default(1, 1, 0, write_scalar, value, 42);
    checkCudaErrors(cudaDeviceSynchronize());

    int host = 0;
    value.view().copy_to(&host);
    require(value.data() == original_allocation,
            "kernel launcher replaced the owning buffer allocation");
    require(host == 42, "kernel launcher wrote into a copied DeviceBuffer instead of the owner");
}

void test_triplet_workspace_preserve()
{
    using BlockMatrix = GIPCTripletMatrix::BlockMatrix;

    auto make_values = [](int count, int base)
    {
        std::vector<BlockMatrix> values(count);
        for(int i = 0; i < count; ++i)
        {
            for(int row = 0; row < 3; ++row)
            {
                for(int col = 0; col < 3; ++col)
                    values[i](row, col) = base + i * 100 + row * 10 + col;
            }
        }
        return values;
    };

    auto require_triplets_equal = [](const std::vector<int>&         rows,
                                     const std::vector<int>&         cols,
                                     const std::vector<BlockMatrix>& values,
                                     const std::vector<int>& expected_rows,
                                     const std::vector<int>& expected_cols,
                                     const std::vector<BlockMatrix>& expected_values,
                                     const char* message)
    {
        require(rows.size() >= expected_rows.size(), message);
        require(cols.size() >= expected_cols.size(), message);
        require(values.size() >= expected_values.size(), message);
        for(size_t i = 0; i < expected_rows.size(); ++i)
        {
            require(rows[i] == expected_rows[i], message);
            require(cols[i] == expected_cols[i], message);
            for(int row = 0; row < 3; ++row)
            {
                for(int col = 0; col < 3; ++col)
                    require(values[i](row, col) == expected_values[i](row, col), message);
            }
        }
    };

    GIPCTripletMatrix triplets;
    triplets.resize(8, 8, 0);
    triplets.resize_conversion_scratch(0);
    require(triplets.triplet_count() == 0 && triplets.triplet_capacity() == 0,
            "triplet workspace must start empty");

    triplets.resize_triplets_discard(4);
    std::vector<int>         rows{0, 1, 2, 3};
    std::vector<int>         cols{7, 6, 5, 4};
    std::vector<BlockMatrix> values = make_values(4, 1000);
    triplets.m_block_row_indices.view().copy_from(rows.data());
    triplets.m_block_col_indices.view().copy_from(cols.data());
    triplets.m_block_values.view().copy_from(values.data());

    triplets.ensure_triplet_capacity(13);
    require(triplets.triplet_count() == 13, "triplet preserve growth must update logical size");
    std::vector<int>         copied_rows;
    std::vector<int>         copied_cols;
    std::vector<BlockMatrix> copied_values;
    triplets.m_block_row_indices.copy_to(copied_rows);
    triplets.m_block_col_indices.copy_to(copied_cols);
    triplets.m_block_values.copy_to(copied_values);
    require_triplets_equal(copied_rows, copied_cols, copied_values, rows, cols, values, "triplet preserve growth lost collision data");

    std::vector<int>         extended_rows(13);
    std::vector<int>         extended_cols(13);
    std::vector<BlockMatrix> extended_values = make_values(13, 5000);
    for(int i = 0; i < 13; ++i)
    {
        extended_rows[i] = 100 + i;
        extended_cols[i] = 300 - i;
    }
    triplets.m_block_row_indices.view().copy_from(extended_rows.data());
    triplets.m_block_col_indices.view().copy_from(extended_cols.data());
    triplets.m_block_values.view().copy_from(extended_values.data());
    triplets.ensure_triplet_capacity(37);
    triplets.m_block_row_indices.copy_to(copied_rows);
    triplets.m_block_col_indices.copy_to(copied_cols);
    triplets.m_block_values.copy_to(copied_values);
    require_triplets_equal(copied_rows,
                           copied_cols,
                           copied_values,
                           extended_rows,
                           extended_cols,
                           extended_values,
                           "second triplet growth lost ABD/FEM staging data");

    triplets.resize_conversion_scratch(0);
    triplets.resize_conversion_scratch(7);
    triplets.resize_conversion_scratch(29);
    require(triplets.m_block_hash_value.size() == 29
                && triplets.m_block_sort_index.size() == 29,
            "conversion scratch did not follow final matrix size");
}

void test_zero_collision_conversion_workspace()
{
    GIPCTripletMatrix triplets;
    triplets.resize(128, 128, 0);
    triplets.resize_conversion_scratch(0);

    // A zero-contact scene can still append and convert ABD body Hessians.
    // The converter writes its sorted staging range immediately after the
    // 640 live blocks, so both triplets and scratch must grow from zero.
    triplets.prepare_conversion_workspace(0, 640, 640);
    require(triplets.triplet_count() == 1280,
            "zero-contact ABD conversion did not allocate its staging range");
    require(triplets.m_block_hash_value.size() == 640
                && triplets.m_block_sort_hash_value.size() == 640
                && triplets.m_block_index.size() == 640
                && triplets.m_block_sort_index.size() == 640
                && triplets.m_block_temp_buffer.size() == 640,
            "zero-contact ABD conversion left scratch at zero capacity");
}

void test_preallocated_triplet_workspace()
{
    GIPCTripletMatrix triplets;
    triplets.resize(64, 64, 0);
    triplets.reserve_triplets(2048);
    triplets.reserve_conversion_scratch(1024);

    require(triplets.triplet_count() == 0,
            "triplet preallocation must not expose uninitialized entries");
    require(triplets.triplet_capacity() >= 2048,
            "triplet value/index staging capacity was not reserved");
    require(triplets.m_block_hash_value.size() == 0
                && triplets.m_block_index.size() == 0,
            "conversion scratch preallocation changed its logical size");
    require(triplets.conversion_scratch_capacity() >= 1024,
            "conversion scratch capacity was not reserved");
}

void test_contact_partition_boundaries()
{
    auto verify = [](int ff_start,
                     int af_start,
                     int fa_start,
                     int aa_start,
                     int total,
                     uint32_t ff_count,
                     uint32_t af_count,
                     uint32_t fa_count,
                     uint32_t aa_count)
    {
        GIPCTripletMatrix triplets;
        triplets.h_fem_fem_contact_start_id = ff_start;
        triplets.h_abd_fem_contact_start_id = af_start;
        triplets.h_fem_abd_contact_start_id = fa_start;
        triplets.h_abd_abd_contact_start_id = aa_start;
        triplets.update_contact_partition_counts(total);
        require(triplets.fem_fem_contact_num == ff_count, "invalid FEM/FEM partition count");
        require(triplets.abd_fem_contact_num == af_count, "invalid ABD/FEM partition count");
        require(triplets.fem_abd_contact_num == fa_count, "invalid FEM/ABD partition count");
        require(triplets.abd_abd_contact_num == aa_count, "invalid ABD/ABD partition count");
    };

    verify(-1, -1, -1, -1, 0, 0, 0, 0, 0);
    verify(-1, 0, -1, -1, 7, 0, 7, 0, 0);
    verify(-1, -1, 0, -1, 9, 0, 0, 9, 0);
    verify(-1, -1, -1, 0, 11, 0, 0, 0, 11);
    verify(0, 2, 5, 8, 13, 2, 3, 3, 5);
    verify(-1, 0, -1, 4, 10, 0, 4, 0, 6);
}

void test_partial_warp_segmental_reduce()
{
    constexpr int n = 35;  // deliberately leaves a three-lane tail warp
    std::vector<int> offsets(n);
    std::vector<double> input(n, 1.0);
    for(int i = 0; i < n; ++i)
        offsets[i] = i < 5 ? 0 : (i < 33 ? 1 : 2);

    cudatool::DeviceBuffer<int> d_offsets(offsets);
    cudatool::DeviceBuffer<double> d_input(input);
    cudatool::DeviceBuffer<double> d_output(3);
    cudatool::parallel::FastSegmentalReduce<>::reduce(
        cudatool::CBufferView<int>(d_offsets.data(), d_offsets.size()),
        cudatool::CBufferView<double>(d_input.data(), d_input.size()),
        d_output.view());

    std::vector<double> output;
    d_output.copy_to(output);
    require(output == std::vector<double>({5.0, 28.0, 2.0}),
            "partial-warp segmented reduction lost or added tail values");
}

void test_mas_going_next_capacity()
{
    constexpr size_t vertex_count = 38'386;
    constexpr size_t level_count  = 4;
    const size_t required = MASPreconditioner::requiredGoingNextCapacity(
        vertex_count, vertex_count, level_count);
    require(required == 38'400 * level_count,
            "MAS hierarchy capacity did not include per-level bank alignment");
    require(required > vertex_count * level_count,
            "MAS regression case no longer exercises the old undersized formula");

    const size_t grouped = MASPreconditioner::requiredGoingNextCapacity(
        1'001, 1'024, 6);
    require(grouped == 1'024 * 6,
            "MAS GROUP capacity did not use the mapped-node domain");
    require(MASPreconditioner::requiredGoingNextCapacity(0, 0, 6) == 0,
            "empty MAS hierarchy must require zero storage");
}

void test_zero_capacity_count_grow_rerun()
{
    constexpr int first_count = 137;

    cudatool::DeviceBuffer<int4>     pairs;
    cudatool::DeviceBuffer<int4>     ccd_pairs;
    cudatool::DeviceBuffer<int>      matrix_indices;
    cudatool::DeviceBuffer<uint32_t> count(1);

    for(;;)
    {
        size_t capacity = std::min(
            {pairs.capacity(), ccd_pairs.capacity(), matrix_indices.capacity()});
        pairs.resize(pairs.capacity());
        ccd_pairs.resize(ccd_pairs.capacity());
        matrix_indices.resize(matrix_indices.capacity());
        count.reset_zero();

        synthetic_detect<<<(first_count + 63) / 64, 64>>>(first_count,
                                                          static_cast<uint32_t>(capacity),
                                                          pairs.data(),
                                                          ccd_pairs.data(),
                                                          matrix_indices.data(),
                                                          count.data());
        uint32_t detected = 0;
        count.view().copy_to(&detected);
        require(detected == first_count, "bounded pass did not preserve the exact count");

        if(detected <= capacity)
        {
            pairs.resize(detected);
            ccd_pairs.resize(detected);
            matrix_indices.resize(detected);
            break;
        }

        pairs.resize_discard(detected);
        ccd_pairs.resize_discard(detected);
        matrix_indices.resize_discard(detected);
    }

    std::vector<int4> first_pairs;
    pairs.copy_to(first_pairs);
    std::vector<int> first_ids;
    first_ids.reserve(first_pairs.size());
    for(const auto& pair : first_pairs)
    {
        require(pair.w == pair.x + 3, "rerun produced an invalid collision pair");
        first_ids.push_back(pair.x);
    }
    std::sort(first_ids.begin(), first_ids.end());
    for(int i = 0; i < first_count; ++i)
        require(first_ids[i] == i, "rerun did not regenerate every collision pair");

    // Full CCD may grow its own output, but must not invalidate the live DCD
    // pairs that line search and MAS still consume.
    constexpr int second_count = 401;
    for(;;)
    {
        size_t capacity = ccd_pairs.capacity();
        ccd_pairs.resize(capacity);
        count.reset_zero();
        synthetic_ccd_detect<<<(second_count + 63) / 64, 64>>>(second_count,
                                                               static_cast<uint32_t>(capacity),
                                                               ccd_pairs.data(),
                                                               count.data());

        uint32_t detected = 0;
        count.view().copy_to(&detected);
        if(detected <= capacity)
        {
            ccd_pairs.resize(detected);
            break;
        }
        ccd_pairs.resize_discard(detected);
    }

    std::vector<int4> preserved_pairs;
    pairs.copy_to(preserved_pairs);
    require(preserved_pairs.size() == first_pairs.size(),
            "full CCD changed the DCD logical range");
    for(size_t i = 0; i < first_pairs.size(); ++i)
        require(preserved_pairs[i].x == first_pairs[i].x
                    && preserved_pairs[i].w == first_pairs[i].w,
                "full CCD growth invalidated DCD pairs");
}

void test_persistent_cub_workspace()
{
    cudatool::DeviceBuffer<int> input;
    cudatool::DeviceBuffer<int> output;

    for(int n : {7, 1025})
    {
        input.resize_discard(n);
        output.resize_discard(n);
        input.fill(1);
        cudatool::DeviceScan().ExclusiveSum(input.data(), output.data(), n);

        std::vector<int> host;
        output.copy_to(host);
        require(host.front() == 0 && host.back() == n - 1,
                "persistent CUB workspace produced an invalid scan");
    }

    cudatool::DeviceBuffer<int> result(1);
    cudatool::DeviceReduce().Sum(input.data(), result.data(), static_cast<int>(input.size()));
    int host_result = -1;
    result.view().copy_to(&host_result);
    require(host_result == static_cast<int>(input.size()),
            "persistent CUB reduction produced an invalid positive-length sum");

    result.fill(17);
    cudatool::DeviceReduce().Sum(input.data(), result.data(), 0);
    host_result = -1;
    result.view().copy_to(&host_result);
    require(host_result == 0, "zero-length CUB reduction did not produce zero");

    cudatool::DeviceBuffer<int> selected_count(1);
    selected_count.fill(17);
    cudatool::DeviceSelect().Flagged(
        input.data(), input.data(), output.data(), selected_count.data(), 0);
    selected_count.view().copy_to(&host_result);
    require(host_result == 0, "zero-length CUB selection did not reset its count");

    cudatool::DeviceRadixSort().SortPairs(
        input.data(), output.data(), input.data(), output.data(), 0);
    cudatool::DeviceScan().ExclusiveSum(input.data(), output.data(), 0);

    cudatool::DeviceBuffer<uint64_t> keys_in(std::vector<uint64_t>{3, 1, 2, 1});
    cudatool::DeviceBuffer<uint64_t> keys_out(4);
    cudatool::DeviceBuffer<uint32_t> values_in(std::vector<uint32_t>{0, 1, 2, 3});
    cudatool::DeviceBuffer<uint32_t> values_out(4);
    cudatool::DeviceRadixSort().SortPairs(
        keys_in.data(), keys_out.data(), values_in.data(), values_out.data(), 4);
    std::vector<uint64_t> sorted_keys;
    std::vector<uint32_t> sorted_values;
    keys_out.copy_to(sorted_keys);
    values_out.copy_to(sorted_values);
    require(sorted_keys == std::vector<uint64_t>({1, 1, 2, 3}),
            "persistent CUB radix sort produced invalid keys");
    require(sorted_values == std::vector<uint32_t>({1, 3, 2, 0}),
            "persistent CUB radix sort did not preserve key/value pairs");
}
}  // namespace

int main()
{
    checkCudaErrors(cudaSetDevice(0));
    test_device_buffer_preserve();
    test_launcher_does_not_copy_owning_buffers();
    test_triplet_workspace_preserve();
    test_zero_collision_conversion_workspace();
    test_preallocated_triplet_workspace();
    test_contact_partition_boundaries();
    test_partial_warp_segmental_reduce();
    test_mas_going_next_capacity();
    test_zero_capacity_count_grow_rerun();
    test_persistent_cub_workspace();
    checkCudaErrors(cudaDeviceSynchronize());
    std::puts("dynamic-memory tests passed");
    return 0;
}
