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

#include "deep_tensor/tensor.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace deep_ros
{

size_t get_dtype_size(DataType dtype)
{
  switch (dtype) {
    case DataType::FLOAT32:
      return sizeof(float);
    case DataType::FLOAT64:
      return sizeof(double);
    case DataType::INT8:
      return sizeof(int8_t);
    case DataType::INT16:
      return sizeof(int16_t);
    case DataType::INT32:
      return sizeof(int32_t);
    case DataType::INT64:
      return sizeof(int64_t);
    case DataType::UINT8:
      return sizeof(uint8_t);
    case DataType::UINT16:
      return sizeof(uint16_t);
    case DataType::UINT32:
      return sizeof(uint32_t);
    case DataType::UINT64:
      return sizeof(uint64_t);
    case DataType::BOOL:
      return sizeof(bool);
    default:
      throw std::invalid_argument("Unknown data type");
  }
}

Tensor::Tensor()
: dtype_(DataType::FLOAT32)
, byte_size_(0)
, data_(nullptr)
, owns_data_(false)
{}

Tensor::Tensor(const std::vector<size_t> & shape, DataType dtype)
: shape_(shape)
, dtype_(dtype)
, owns_data_(true)
{
  calculate_strides();

  size_t total_elements = std::accumulate(shape_.begin(), shape_.end(), 1UL, std::multiplies<size_t>());
  byte_size_ = total_elements * get_dtype_size(dtype_);

  allocate_memory();
}

Tensor::Tensor(void * data, const std::vector<size_t> & shape, DataType dtype)
: shape_(shape)
, dtype_(dtype)
, data_(data)
, owns_data_(false)
{
  calculate_strides();

  size_t total_elements = std::accumulate(shape_.begin(), shape_.end(), 1UL, std::multiplies<size_t>());
  byte_size_ = total_elements * get_dtype_size(dtype_);
}

Tensor::~Tensor()
{
  deallocate_memory();
}

Tensor::Tensor(const Tensor & other)
: shape_(other.shape_)
, strides_(other.strides_)
, dtype_(other.dtype_)
, byte_size_(other.byte_size_)
, owns_data_(true)
{
  allocate_memory();
  if (other.data_ && data_) {
    std::memcpy(data_, other.data_, byte_size_);
  }
}

Tensor & Tensor::operator=(const Tensor & other)
{
  if (this != &other) {
    deallocate_memory();

    shape_ = other.shape_;
    strides_ = other.strides_;
    dtype_ = other.dtype_;
    byte_size_ = other.byte_size_;
    owns_data_ = true;

    allocate_memory();
    if (other.data_ && data_) {
      std::memcpy(data_, other.data_, byte_size_);
    }
  }
  return *this;
}

Tensor::Tensor(Tensor && other) noexcept
: shape_(std::move(other.shape_))
, strides_(std::move(other.strides_))
, dtype_(other.dtype_)
, byte_size_(other.byte_size_)
, data_(other.data_)
, owns_data_(other.owns_data_)
{
  other.data_ = nullptr;
  other.owns_data_ = false;
  other.byte_size_ = 0;
}

Tensor & Tensor::operator=(Tensor && other) noexcept
{
  if (this != &other) {
    deallocate_memory();

    shape_ = std::move(other.shape_);
    strides_ = std::move(other.strides_);
    dtype_ = other.dtype_;
    byte_size_ = other.byte_size_;
    data_ = other.data_;
    owns_data_ = other.owns_data_;

    other.data_ = nullptr;
    other.owns_data_ = false;
    other.byte_size_ = 0;
  }
  return *this;
}

void Tensor::calculate_strides()
{
  strides_.resize(shape_.size());
  if (shape_.empty()) return;

  strides_.back() = get_dtype_size(dtype_);
  for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
    strides_[i] = strides_[i + 1] * shape_[i + 1];
  }
}

void Tensor::allocate_memory()
{
  if (byte_size_ > 0 && owns_data_) {
    data_ = std::aligned_alloc(32, byte_size_);
    if (!data_) {
      throw std::bad_alloc();
    }
  }
}

void Tensor::deallocate_memory()
{
  if (owns_data_ && data_) {
    std::free(data_);
    data_ = nullptr;
  }
}

Tensor Tensor::reshape(const std::vector<size_t> & new_shape) const
{
  size_t new_total = std::accumulate(new_shape.begin(), new_shape.end(), 1UL, std::multiplies<size_t>());
  size_t current_total = std::accumulate(shape_.begin(), shape_.end(), 1UL, std::multiplies<size_t>());

  if (new_total != current_total) {
    throw std::invalid_argument("Cannot reshape tensor: total size mismatch");
  }

  if (!is_contiguous()) {
    throw std::runtime_error("Cannot reshape non-contiguous tensor");
  }

  return Tensor(data_, new_shape, dtype_);
}

size_t Tensor::size() const
{
  return std::accumulate(shape_.begin(), shape_.end(), 1UL, std::multiplies<size_t>());
}

bool Tensor::is_contiguous() const
{
  if (shape_.empty()) return true;

  size_t expected_stride = get_dtype_size(dtype_);
  for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
    if (strides_[i] != expected_stride) {
      return false;
    }
    expected_stride *= shape_[i];
  }
  return true;
}

}  // namespace deep_ros
