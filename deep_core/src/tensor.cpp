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
#include <cstring>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace deep_ros
{

namespace
{
size_t get_dtype_size(DataType dtype)
{
  switch (dtype) {
    case DataType::FLOAT32:
      return sizeof(float);
    case DataType::FLOAT64:
      return sizeof(double);
    case DataType::INT32:
      return sizeof(int32_t);
    case DataType::INT64:
      return sizeof(int64_t);
    case DataType::UINT8:
      return sizeof(uint8_t);
    case DataType::BOOL:
      return sizeof(bool);
    default:
      throw TensorError(std::format("Unknown data type: {}", static_cast<int>(dtype)));
  }
}
}  // namespace

Tensor::Tensor(std::vector<int64_t> shape, DataType dtype, size_t size_bytes)
: shape_(std::move(shape))
, dtype_(dtype)
, size_bytes_(size_bytes)
{
  if (size_bytes_ > 0) {
    data_ = std::make_unique<uint8_t[]>(size_bytes_);
  }
}

std::unique_ptr<Tensor> Tensor::create_float32(std::vector<int64_t> shape, std::vector<float> data)
{
  validate_shape_and_data(shape, data.size());

  size_t size_bytes = data.size() * sizeof(float);
  auto tensor = std::unique_ptr<Tensor>(new Tensor(shape, DataType::FLOAT32, size_bytes));

  std::memcpy(tensor->data_.get(), data.data(), size_bytes);
  return tensor;
}

std::unique_ptr<Tensor> Tensor::create_int32(std::vector<int64_t> shape, std::vector<int32_t> data)
{
  validate_shape_and_data(shape, data.size());

  size_t size_bytes = data.size() * sizeof(int32_t);
  auto tensor = std::unique_ptr<Tensor>(new Tensor(shape, DataType::INT32, size_bytes));

  std::memcpy(tensor->data_.get(), data.data(), size_bytes);
  return tensor;
}

std::unique_ptr<Tensor> Tensor::clone() const
{
  if (!data_) {
    throw TensorError("Cannot clone tensor with null data");
  }

  auto cloned = std::unique_ptr<Tensor>(new Tensor(shape_, dtype_, size_bytes_));
  std::memcpy(cloned->data_.get(), data_.get(), size_bytes_);
  return cloned;
}

size_t Tensor::element_count() const noexcept
{
  if (shape_.empty()) return 0;
  return std::accumulate(shape_.begin(), shape_.end(), 1LL, std::multiplies<int64_t>());
}

template <>
bool Tensor::validate_type<float>() const
{
  return dtype_ == DataType::FLOAT32;
}

template <>
bool Tensor::validate_type<double>() const
{
  return dtype_ == DataType::FLOAT64;
}

template <>
bool Tensor::validate_type<int32_t>() const
{
  return dtype_ == DataType::INT32;
}

template <>
bool Tensor::validate_type<int64_t>() const
{
  return dtype_ == DataType::INT64;
}

template <>
bool Tensor::validate_type<uint8_t>() const
{
  return dtype_ == DataType::UINT8;
}

template <>
bool Tensor::validate_type<bool>() const
{
  return dtype_ == DataType::BOOL;
}

void Tensor::validate_shape_and_data(const std::vector<int64_t> & shape, size_t data_size)
{
  if (shape.empty()) {
    throw TensorError("Tensor shape cannot be empty");
  }

  for (const auto & dim : shape) {
    if (dim <= 0) {
      throw TensorError(std::format("Invalid shape dimension: {}", dim));
    }
  }

  size_t expected_size = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
  if (data_size != expected_size) {
    throw TensorError(std::format("Data size {} doesn't match shape size {}", data_size, expected_size));
  }
}

}  // namespace deep_ros
