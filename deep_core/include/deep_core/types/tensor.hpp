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

#include <filesystem>
#include <format>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace deep_ros
{

enum class DataType : uint8_t
{
  FLOAT32,
  FLOAT64,
  INT32,
  INT64,
  UINT8,
  BOOL
};

class TensorError : public std::runtime_error
{
public:
  explicit TensorError(const std::string & message)
  : std::runtime_error(message)
  {}
};

class Tensor
{
public:
  // Factory methods for type safety
  static std::unique_ptr<Tensor> create_float32(std::vector<int64_t> shape, std::vector<float> data);

  static std::unique_ptr<Tensor> create_int32(std::vector<int64_t> shape, std::vector<int32_t> data);

  // Move semantics for efficiency
  Tensor(Tensor &&) noexcept = default;
  Tensor & operator=(Tensor &&) noexcept = default;

  // Prevent expensive copying with clear error message
  Tensor(const Tensor &) = delete;
  Tensor & operator=(const Tensor &) = delete;
  std::unique_ptr<Tensor> clone() const;

  // Const-correct accessors with detailed error messages
  const std::vector<int64_t> & shape() const noexcept
  {
    return shape_;
  }

  DataType dtype() const noexcept
  {
    return dtype_;
  }

  const void * data() const
  {
    if (!data_) {
      throw TensorError("Tensor data is null - tensor may have been moved or uninitialized");
    }
    return data_.get();
  }

  size_t size_bytes() const noexcept
  {
    return size_bytes_;
  }

  // Utility methods with validation
  size_t element_count() const noexcept;

  bool is_valid() const noexcept
  {
    return data_ != nullptr;
  }

  // Bounds-checked typed data access
  template <typename T>
  const T * typed_data() const
  {
    if (!data_) {
      throw TensorError("Cannot access typed data - tensor data is null");
    }

    if (!validate_type<T>()) {
      throw TensorError(
        std::format("Type mismatch - requested type doesn't match tensor dtype {}", static_cast<int>(dtype_)));
    }

    return reinterpret_cast<const T *>(data_.get());
  }

private:
  Tensor(std::vector<int64_t> shape, DataType dtype, size_t size_bytes);

  template <typename T>
  bool validate_type() const;

  static void validate_shape_and_data(const std::vector<int64_t> & shape, size_t data_size);

  std::vector<int64_t> shape_;
  DataType dtype_;
  std::unique_ptr<uint8_t[]> data_;
  size_t size_bytes_;
};

}  // namespace deep_ros
