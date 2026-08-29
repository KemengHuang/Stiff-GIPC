#pragma once

#include <algorithm>
#include <cstdlib>
#include <cuda_tools/cuda_buffer_view.h>
#include <iostream>
#include <limits>
#include "cuda_tools/cuda_tools.h"
#include "Eigen/Eigen"

//#define SymGH
#ifdef SymGH
#define M12_Off 10
#define M9_Off 6
#define M6_Off 3
#else
#define M12_Off 16
#define M9_Off 9
#define M6_Off 4
#endif


class GIPCTripletMatrix
{
  public:
    using BlockMatrix = Eigen::Matrix<double, 3, 3>;
    //using EntryValueType = T;
    //int Dimenstion              = M;
  public:
    cudatool::DeviceBuffer<BlockMatrix> m_block_values;
    cudatool::DeviceBuffer<int>         m_block_row_indices;
    cudatool::DeviceBuffer<int>         m_block_col_indices;
    cudatool::DeviceBuffer<uint64_t>    m_block_hash_value;
    cudatool::DeviceBuffer<uint64_t>    m_block_sort_hash_value;
    cudatool::DeviceBuffer<uint32_t>    m_block_index;
    cudatool::DeviceBuffer<uint32_t>    m_block_sort_index;
    cudatool::DeviceBuffer<uint32_t>    m_block_temp_buffer;
    int                                 m_block_rows = 0;
    int                                 m_block_cols = 0;

  public:
    GIPCTripletMatrix()                                    = default;
    ~GIPCTripletMatrix()                                   = default;
    GIPCTripletMatrix(const GIPCTripletMatrix&)            = delete;
    GIPCTripletMatrix(GIPCTripletMatrix&&)                 = default;
    GIPCTripletMatrix& operator=(const GIPCTripletMatrix&) = delete;
    GIPCTripletMatrix& operator=(GIPCTripletMatrix&&)      = default;

    void reshape(int row, int col)
    {
        m_block_rows = row;
        m_block_cols = col;
    }

    void resize_triplets(size_t nonzero_count)
    {
        m_block_values.resize(nonzero_count);
        m_block_row_indices.resize(nonzero_count);
        m_block_col_indices.resize(nonzero_count);
    }

    void resize_triplets_discard(size_t nonzero_count)
    {
        m_block_values.resize_discard(nonzero_count);
        m_block_row_indices.resize_discard(nonzero_count);
        m_block_col_indices.resize_discard(nonzero_count);
    }

    void reserve_triplets(size_t nonzero_count)
    {
        m_block_values.reserve(nonzero_count);
        m_block_row_indices.reserve(nonzero_count);
        m_block_col_indices.reserve(nonzero_count);
    }

    // Grow the value/row/col buffers while preserving their complete logical
    // range, and make nonzero_count the new writable range.
    void ensure_triplet_capacity(size_t nonzero_count)
    {
        m_block_values.resize_preserve(nonzero_count);
        m_block_row_indices.resize_preserve(nonzero_count);
        m_block_col_indices.resize_preserve(nonzero_count);
    }

    void resize(int row, int col, size_t nonzero_count)
    {
        reshape(row, col);
        resize_triplets(nonzero_count);
    }

    void resize_conversion_scratch(size_t nonzero_count)
    {
        m_block_hash_value.resize_discard(nonzero_count);
        m_block_sort_hash_value.resize_discard(nonzero_count);
        m_block_index.resize_discard(nonzero_count);
        m_block_sort_index.resize_discard(nonzero_count);
        m_block_temp_buffer.resize_discard(nonzero_count);
    }

    void reserve_conversion_scratch(size_t nonzero_count)
    {
        m_block_hash_value.reserve(nonzero_count);
        m_block_sort_hash_value.reserve(nonzero_count);
        m_block_index.reserve(nonzero_count);
        m_block_sort_index.reserve(nonzero_count);
        m_block_temp_buffer.reserve(nonzero_count);
    }

    size_t conversion_scratch_capacity() const
    {
        return std::min({m_block_hash_value.capacity(),
                         m_block_sort_hash_value.capacity(),
                         m_block_index.capacity(),
                         m_block_sort_index.capacity(),
                         m_block_temp_buffer.capacity()});
    }

    void prepare_conversion_workspace(size_t input_start, size_t nonzero_count, size_t output_start)
    {
        if(input_start > std::numeric_limits<size_t>::max() - nonzero_count
           || output_start > std::numeric_limits<size_t>::max() - nonzero_count)
        {
            std::cerr << "Triplet conversion range overflow." << std::endl;
            std::abort();
        }
        const size_t required_triplets = std::max(
            {m_block_values.size(), input_start + nonzero_count, output_start + nonzero_count});
        ensure_triplet_capacity(required_triplets);
        resize_conversion_scratch(nonzero_count);
    }

    void reset_zero()
    {
        m_block_values.reset_zero();
        m_block_row_indices.reset_zero();
        m_block_col_indices.reset_zero();
    }

    void update_hash_value(int fem_offset);

  private:
    template <typename T>
    static T* offset_pointer(T* pointer, int offset)
    {
        if(offset < 0)
        {
            std::cerr << "Negative triplet-buffer offset: " << offset << std::endl;
            std::abort();
        }
        return pointer ? pointer + offset : nullptr;
    }

  public:
    auto block_values(int offset = 0) { return offset_pointer(m_block_values.data(), offset); }
    auto block_values(int offset = 0) const
    {
        return offset_pointer(m_block_values.data(), offset);
    }
    auto block_row_indices(int offset = 0)
    {
        return offset_pointer(m_block_row_indices.data(), offset);
    }
    auto block_row_indices(int offset = 0) const
    {
        return offset_pointer(m_block_row_indices.data(), offset);
    }
    auto block_col_indices(int offset = 0)
    {
        return offset_pointer(m_block_col_indices.data(), offset);
    }
    auto block_col_indices(int offset = 0) const
    {
        return offset_pointer(m_block_col_indices.data(), offset);
    }
    auto block_hash_value(int offset = 0)
    {
        return offset_pointer(m_block_hash_value.data(), offset);
    }
    auto block_hash_value(int offset = 0) const
    {
        return offset_pointer(m_block_hash_value.data(), offset);
    }

    auto block_sort_hash_value(int offset = 0)
    {
        return offset_pointer(m_block_sort_hash_value.data(), offset);
    }
    auto block_sort_hash_value(int offset = 0) const
    {
        return offset_pointer(m_block_sort_hash_value.data(), offset);
    }

    auto block_temp_buffer(int offset = 0)
    {
        return offset_pointer(m_block_temp_buffer.data(), offset);
    }
    auto block_temp_buffer(int offset = 0) const
    {
        return offset_pointer(m_block_temp_buffer.data(), offset);
    }

    auto block_index(int offset = 0) { return offset_pointer(m_block_index.data(), offset); }
    auto block_index(int offset = 0) const
    {
        return offset_pointer(m_block_index.data(), offset);
    }

    auto block_sort_index(int offset = 0)
    {
        return offset_pointer(m_block_sort_index.data(), offset);
    }
    auto block_sort_index(int offset = 0) const
    {
        return offset_pointer(m_block_sort_index.data(), offset);
    }

    auto block_rows() const { return m_block_rows; }
    auto block_cols() const { return m_block_cols; }
    auto triplet_count() const { return m_block_values.size(); }
    auto triplet_capacity() const { return m_block_values.capacity(); }

    void clear()
    {
        m_block_rows = 0;
        m_block_cols = 0;
        m_block_values.clear();
        m_block_row_indices.clear();
        m_block_col_indices.clear();
    }
    int                         global_triplet_offset           = 0;
    int                         global_collision_triplet_offset = 0;
    cudatool::DeviceBuffer<int> d_abd_abd_contact_start_id;
    cudatool::DeviceBuffer<int> d_abd_fem_contact_start_id;
    cudatool::DeviceBuffer<int> d_fem_abd_contact_start_id;
    cudatool::DeviceBuffer<int> d_fem_fem_contact_start_id;
    cudatool::DeviceBuffer<int> d_unique_key_number;

    void init_var()
    {
        d_abd_abd_contact_start_id.resize(1);
        d_abd_fem_contact_start_id.resize(1);
        d_fem_abd_contact_start_id.resize(1);
        d_fem_fem_contact_start_id.resize(1);
        d_unique_key_number.resize(1);
    }

    int h_abd_abd_contact_start_id = -1;
    int h_abd_fem_contact_start_id = -1;
    int h_fem_abd_contact_start_id = -1;
    int h_fem_fem_contact_start_id = -1;
    int h_unique_key_number        = 0;

    uint32_t abd_abd_contact_num = 0;
    uint32_t abd_fem_contact_num = 0;
    uint32_t fem_fem_contact_num = 0;
    uint32_t fem_abd_contact_num = 0;

    // Convert the optional starts produced from sorted contact hashes
    // (0=fem/fem, 1=abd/fem, 2=fem/abd, 3=abd/abd) into exact counts.
    // A present partition is allowed to start at zero.
    void update_contact_partition_counts(int total_count)
    {
        if(total_count < 0)
        {
            std::cerr << "Negative contact triplet count." << std::endl;
            std::abort();
        }

        int starts[4] = {h_fem_fem_contact_start_id,
                         h_abd_fem_contact_start_id,
                         h_fem_abd_contact_start_id,
                         h_abd_abd_contact_start_id};
        uint32_t counts[4] = {};
        int previous_start = -1;
        for(int type = 0; type < 4; ++type)
        {
            const int start = starts[type];
            if(start < -1 || start > total_count || (start >= 0 && start < previous_start))
            {
                std::cerr << "Invalid sorted contact partition start for type " << type
                          << ": " << start << " (total " << total_count << ")" << std::endl;
                std::abort();
            }
            if(start < 0)
                continue;

            previous_start = start;
            int end = total_count;
            for(int next_type = type + 1; next_type < 4; ++next_type)
            {
                if(starts[next_type] >= 0)
                {
                    end = starts[next_type];
                    break;
                }
            }
            if(end < start)
            {
                std::cerr << "Contact partition ends before it starts." << std::endl;
                std::abort();
            }
            counts[type] = static_cast<uint32_t>(end - start);
        }

        const uint64_t counted = static_cast<uint64_t>(counts[0]) + counts[1]
                                 + counts[2] + counts[3];
        if(counted != static_cast<uint64_t>(total_count))
        {
            std::cerr << "Contact partitions cover " << counted << " of " << total_count
                      << " sorted triplets." << std::endl;
            std::abort();
        }

        fem_fem_contact_num = counts[0];
        abd_fem_contact_num = counts[1];
        fem_abd_contact_num = counts[2];
        abd_abd_contact_num = counts[3];

        h_fem_fem_contact_start_id = 0;
        h_abd_fem_contact_start_id = static_cast<int>(fem_fem_contact_num);
        h_fem_abd_contact_start_id =
            h_abd_fem_contact_start_id + static_cast<int>(abd_fem_contact_num);
        h_abd_abd_contact_start_id =
            h_fem_abd_contact_start_id + static_cast<int>(fem_abd_contact_num);
    }
};
