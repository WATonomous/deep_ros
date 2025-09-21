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

#include "deep_core/types/tensor.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
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

TensorPtr::TensorPtr()
: dtype_(DataType::FLOAT32)
, byte_size_(0)
, data_(nullptr)
, is_view_(false)
, allocator_(nullptr)
{}

TensorPtr::TensorPtr(const std::vector<size_t> & shape, DataType dtype)
: shape_(shape)
, dtype_(dtype)
, is_view_(false)
, allocator_(nullptr)
{
  throw std::runtime_error(
    "TensorPtr construction requires an allocator. Use TensorPtr(shape, dtype, allocator) instead.");
}

TensorPtr::TensorPtr(
  const std::vector<size_t> & shape, DataType dtype, std::shared_ptr<BackendMemoryAllocator> allocator)
: shape_(shape)
, dtype_(dtype)
, is_view_(false)
, allocator_(allocator)
{
  if (shape_.empty()) {
    throw std::invalid_argument("Tensor shape cannot be empty");
  }

  calculate_strides();

  size_t total_elements = std::accumulate(shape_.begin(), shape_.end(), 1UL, std::multiplies<size_t>());
  byte_size_ = total_elements * get_dtype_size(dtype_);

  allocate_memory();
}

TensorPtr::TensorPtr(void * data, const std::vector<size_t> & shape, DataType dtype)
: shape_(shape)
, dtype_(dtype)
, data_(data)
, is_view_(false)
, allocator_(nullptr)
{
  if (shape_.empty()) {
    throw std::invalid_argument("Tensor shape cannot be empty");
  }

  calculate_strides();

  size_t total_elements = std::accumulate(shape_.begin(), shape_.end(), 1UL, std::multiplies<size_t>());
  byte_size_ = total_elements * get_dtype_size(dtype_);
}

TensorPtr::~TensorPtr()
{
  deallocate_memory();
}

TensorPtr::TensorPtr(const TensorPtr & other)
: shape_(other.shape_)
, strides_(other.strides_)
, dtype_(other.dtype_)
, byte_size_(other.byte_size_)
, is_view_(true)
, allocator_(other.allocator_)
{
  allocate_memory();
  if (other.data_ && data_) {
    if (allocator_ && other.allocator_ && allocator_ == other.allocator_) {
      allocator_->copy_device_to_device(data_, other.data_, byte_size_);
    } else if (allocator_) {
      allocator_->copy_from_host(data_, other.data_, byte_size_);
    } else {
      std::memcpy(data_, other.data_, byte_size_);
    }
  }
}

TensorPtr & TensorPtr::operator=(const TensorPtr & other)
{
  if (this != &other) {
    deallocate_memory();

    shape_ = other.shape_;
    strides_ = other.strides_;
    dtype_ = other.dtype_;
    byte_size_ = other.byte_size_;
    is_view_ = true;
    allocator_ = other.allocator_;

    allocate_memory();
    if (other.data_ && data_) {
      if (allocator_ && other.allocator_ && allocator_ == other.allocator_) {
        allocator_->copy_device_to_device(data_, other.data_, byte_size_);
      } else if (allocator_) {
        allocator_->copy_from_host(data_, other.data_, byte_size_);
      } else {
        std::memcpy(data_, other.data_, byte_size_);
      }
    }
  }
  return *this;
}

TensorPtr::TensorPtr(TensorPtr && other) noexcept
: shape_(std::move(other.shape_))
, strides_(std::move(other.strides_))
, dtype_(other.dtype_)
, byte_size_(other.byte_size_)
, data_(other.data_)
, is_view_(other.is_view_)
, allocator_(std::move(other.allocator_))
{
  other.data_ = nullptr;
  other.is_view_ = false;
  other.byte_size_ = 0;
  other.allocator_ = nullptr;
}

TensorPtr & TensorPtr::operator=(TensorPtr && other) noexcept
{
  if (this != &other) {
    deallocate_memory();

    shape_ = std::move(other.shape_);
    strides_ = std::move(other.strides_);
    dtype_ = other.dtype_;
    byte_size_ = other.byte_size_;
    data_ = other.data_;
    is_view_ = other.is_view_;
    allocator_ = std::move(other.allocator_);

    other.data_ = nullptr;
    other.is_view_ = false;
    other.byte_size_ = 0;
    other.allocator_ = nullptr;
  }
  return *this;
}

void TensorPtr::calculate_strides()
{
  strides_.resize(shape_.size());
  if (shape_.empty()) return;

  strides_.back() = get_dtype_size(dtype_);
  for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
    strides_[i] = strides_[i + 1] * shape_[i + 1];
  }
}

void TensorPtr::allocate_memory()
{
  if (byte_size_ > 0 && !is_view_) {
    if (allocator_) {
      data_ = allocator_->allocate(byte_size_);
    } else {
      data_ = std::aligned_alloc(32, byte_size_);
    }
    if (!data_) {
      throw std::bad_alloc();
    }
  }
}

void TensorPtr::deallocate_memory()
{
  if (!is_view_ && data_) {
    if (allocator_) {
      allocator_->deallocate(data_);
    } else {
      std::free(data_);
    }
    data_ = nullptr;
  }
}

TensorPtr TensorPtr::reshape(const std::vector<size_t> & new_shape) const
{
  size_t new_total = std::accumulate(new_shape.begin(), new_shape.end(), 1UL, std::multiplies<size_t>());
  size_t current_total = std::accumulate(shape_.begin(), shape_.end(), 1UL, std::multiplies<size_t>());

  if (new_total != current_total) {
    throw std::invalid_argument("Cannot reshape tensor: total size mismatch");
  }

  if (!is_contiguous()) {
    throw std::runtime_error("Cannot reshape non-contiguous tensor");
  }

  return TensorPtr(data_, new_shape, dtype_);
}

size_t TensorPtr::size() const
{
  return std::accumulate(shape_.begin(), shape_.end(), 1UL, std::multiplies<size_t>());
}

bool TensorPtr::is_contiguous() const
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
