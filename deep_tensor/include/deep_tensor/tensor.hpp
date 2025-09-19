// Copyright (c) 2025-present WATonomous. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace deep_ros
{

/**
 * @brief Supported data types for tensor elements
 */
enum class DataType : uint8_t
{
  FLOAT32 = 1,
  FLOAT64 = 2,
  INT8 = 3,
  INT16 = 4,
  INT32 = 5,
  INT64 = 6,
  UINT8 = 7,
  UINT16 = 8,
  UINT32 = 9,
  UINT64 = 10,
  BOOL = 11
};

/**
 * @brief Get the size in bytes of a data type
 * @param dtype The data type
 * @return Size in bytes
 */
size_t get_dtype_size(DataType dtype);

/**
 * @brief A multi-dimensional array container with automatic memory management
 *
 * The Tensor class provides a lightweight container for multi-dimensional arrays
 * with support for different data types. It handles memory allocation automatically
 * and supports both owned and borrowed memory patterns.
 */
class Tensor
{
public:
  /**
   * @brief Default constructor - creates an empty tensor
   */
  Tensor();

  /**
   * @brief Create a new tensor with specified shape and data type
   * @param shape Dimensions of the tensor
   * @param dtype Data type of tensor elements
   */
  Tensor(const std::vector<size_t> & shape, DataType dtype);

  /**
   * @brief Wrap existing data in a tensor (non-owning)
   * @param data Pointer to existing data
   * @param shape Dimensions of the tensor
   * @param dtype Data type of tensor elements
   */
  Tensor(void * data, const std::vector<size_t> & shape, DataType dtype);

  /**
   * @brief Destructor - frees owned memory
   */
  ~Tensor();

  /**
   * @brief Copy constructor - creates a deep copy
   */
  Tensor(const Tensor & other);

  /**
   * @brief Copy assignment - creates a deep copy
   */
  Tensor & operator=(const Tensor & other);

  /**
   * @brief Move constructor
   */
  Tensor(Tensor && other) noexcept;

  /**
   * @brief Move assignment
   */
  Tensor & operator=(Tensor && other) noexcept;

  /**
   * @brief Get tensor dimensions
   * @return Vector of dimension sizes
   */
  const std::vector<size_t> & shape() const
  {
    return shape_;
  }

  /**
   * @brief Get memory strides for each dimension
   * @return Vector of stride sizes in bytes
   */
  const std::vector<size_t> & strides() const
  {
    return strides_;
  }

  /**
   * @brief Get number of dimensions
   * @return Number of dimensions
   */
  size_t rank() const
  {
    return shape_.size();
  }

  /**
   * @brief Get data type of tensor elements
   * @return Data type enum
   */
  DataType dtype() const
  {
    return dtype_;
  }

  /**
   * @brief Get total size of tensor data in bytes
   * @return Size in bytes
   */
  size_t byte_size() const
  {
    return byte_size_;
  }

  /**
   * @brief Get raw data pointer
   * @return Pointer to tensor data
   */
  void * data()
  {
    return data_;
  }

  /**
   * @brief Get raw data pointer (const)
   * @return Const pointer to tensor data
   */
  const void * data() const
  {
    return data_;
  }

  /**
   * @brief Get typed data pointer
   * @tparam T Data type to cast to
   * @return Typed pointer to tensor data
   */
  template <typename T>
  T * data_as()
  {
    return static_cast<T *>(data_);
  }

  /**
   * @brief Get typed data pointer (const)
   * @tparam T Data type to cast to
   * @return Const typed pointer to tensor data
   */
  template <typename T>
  const T * data_as() const
  {
    return static_cast<const T *>(data_);
  }

  /**
   * @brief Create a new tensor with different shape but same data
   * @param new_shape New dimensions (total size must match)
   * @return New tensor view with different shape
   */
  Tensor reshape(const std::vector<size_t> & new_shape) const;

  /**
   * @brief Get total number of elements
   * @return Number of elements
   */
  size_t size() const;

  /**
   * @brief Check if tensor data is stored contiguously in memory
   * @return True if contiguous, false otherwise
   */
  bool is_contiguous() const;

private:
  std::vector<size_t> shape_;
  std::vector<size_t> strides_;
  DataType dtype_;
  size_t byte_size_;
  void * data_;
  bool owns_data_;

  void calculate_strides();
  void allocate_memory();
  void deallocate_memory();
};

}  // namespace deep_ros
