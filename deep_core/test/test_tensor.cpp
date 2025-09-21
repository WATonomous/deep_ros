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

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "deep_core/types/tensor.hpp"

namespace deep_ros
{
namespace test
{

class MockMemoryAllocator : public BackendMemoryAllocator
{
public:
  void * allocate(size_t bytes) override
  {
    if (bytes == 0) return nullptr;
    return std::malloc(bytes);
  }

  void deallocate(void * ptr) override
  {
    if (ptr) {
      std::free(ptr);
    }
  }

  void copy_from_host(void * dst, const void * src, size_t bytes) override
  {
    std::memcpy(dst, src, bytes);
  }

  void copy_to_host(void * dst, const void * src, size_t bytes) override
  {
    std::memcpy(dst, src, bytes);
  }

  void copy_device_to_device(void * dst, const void * src, size_t bytes) override
  {
    std::memcpy(dst, src, bytes);
  }

  bool is_device_memory() const override
  {
    return false;
  }

  std::string device_name() const override
  {
    return "mock_device";
  }
};

}  // namespace test
}  // namespace deep_ros

TEST_CASE("deep_ros::Tensor default constructor", "[tensor]")
{
  deep_ros::Tensor tensor;

  REQUIRE(tensor.rank() == 0);
  REQUIRE(tensor.byte_size() == 0);
  REQUIRE(tensor.dtype() == deep_ros::DataType::FLOAT32);
  REQUIRE(tensor.data() == nullptr);
}

TEST_CASE("deep_ros::Tensor shape constructor throws without allocator", "[tensor]")
{
  std::vector<size_t> shape = {2, 3};
  REQUIRE_THROWS_AS(deep_ros::Tensor(shape, deep_ros::DataType::FLOAT32), std::runtime_error);
}

TEST_CASE("deep_ros::Tensor shape constructor with allocator", "[tensor]")
{
  auto allocator = std::make_shared<deep_ros::test::MockMemoryAllocator>();

  SECTION("2D tensor")
  {
    std::vector<size_t> shape = {2, 3};
    deep_ros::Tensor tensor(shape, deep_ros::DataType::FLOAT32, allocator);

    REQUIRE(tensor.shape() == shape);
    REQUIRE(tensor.rank() == 2);
    REQUIRE(tensor.size() == 6);
    REQUIRE(tensor.byte_size() == 6 * sizeof(float));
    REQUIRE(tensor.dtype() == deep_ros::DataType::FLOAT32);
    REQUIRE(tensor.data() != nullptr);
  }

  SECTION("3D tensor with different data type")
  {
    std::vector<size_t> shape = {2, 3, 4};
    deep_ros::Tensor tensor(shape, deep_ros::DataType::INT32, allocator);

    REQUIRE(tensor.shape() == shape);
    REQUIRE(tensor.rank() == 3);
    REQUIRE(tensor.size() == 24);
    REQUIRE(tensor.byte_size() == 24 * sizeof(int32_t));
    REQUIRE(tensor.dtype() == deep_ros::DataType::INT32);
  }
}

TEST_CASE("deep_ros::Tensor external data constructor", "[tensor]")
{
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<size_t> shape = {2, 2};

  deep_ros::Tensor tensor(data.data(), shape, deep_ros::DataType::FLOAT32);

  REQUIRE(tensor.shape() == shape);
  REQUIRE(tensor.rank() == 2);
  REQUIRE(tensor.size() == 4);
  REQUIRE(tensor.data() == data.data());
  REQUIRE(tensor.dtype() == deep_ros::DataType::FLOAT32);
}

TEST_CASE("deep_ros::Tensor copy constructor", "[tensor]")
{
  auto allocator = std::make_shared<deep_ros::test::MockMemoryAllocator>();
  std::vector<size_t> shape = {2, 3};
  deep_ros::Tensor original(shape, deep_ros::DataType::INT32, allocator);

  // Fill with test data
  auto int_data = original.data_as<int32_t>();
  for (size_t i = 0; i < original.size(); ++i) {
    int_data[i] = static_cast<int32_t>(i + 1);
  }

  deep_ros::Tensor copy(original);

  REQUIRE(copy.shape() == original.shape());
  REQUIRE(copy.dtype() == original.dtype());
  REQUIRE(copy.byte_size() == original.byte_size());
  REQUIRE(copy.data() != original.data());  // Different memory

  // Check data was copied correctly
  auto copy_data = copy.data_as<int32_t>();
  for (size_t i = 0; i < copy.size(); ++i) {
    REQUIRE(copy_data[i] == int_data[i]);
  }
}

TEST_CASE("deep_ros::Tensor move constructor", "[tensor]")
{
  auto allocator = std::make_shared<deep_ros::test::MockMemoryAllocator>();
  std::vector<size_t> shape = {3, 4};
  deep_ros::Tensor original(shape, deep_ros::DataType::FLOAT64, allocator);
  void * original_data = original.data();

  deep_ros::Tensor moved(std::move(original));

  REQUIRE(moved.shape() == shape);
  REQUIRE(moved.dtype() == deep_ros::DataType::FLOAT64);
  REQUIRE(moved.data() == original_data);
  REQUIRE(original.data() == nullptr);
  REQUIRE(original.byte_size() == 0);
}

TEST_CASE("deep_ros::Tensor data_as template method", "[tensor]")
{
  auto allocator = std::make_shared<deep_ros::test::MockMemoryAllocator>();
  std::vector<size_t> shape = {2, 2};
  deep_ros::Tensor tensor(shape, deep_ros::DataType::FLOAT32, allocator);

  auto float_ptr = tensor.data_as<float>();
  REQUIRE(float_ptr != nullptr);

  // Test writing and reading
  float_ptr[0] = 1.5f;
  float_ptr[1] = 2.5f;

  REQUIRE(float_ptr[0] == 1.5f);
  REQUIRE(float_ptr[1] == 2.5f);
}

TEST_CASE("deep_ros::Tensor reshape", "[tensor]")
{
  auto allocator = std::make_shared<deep_ros::test::MockMemoryAllocator>();
  std::vector<size_t> shape = {2, 6};
  deep_ros::Tensor tensor(shape, deep_ros::DataType::UINT8, allocator);

  SECTION("Valid reshape")
  {
    std::vector<size_t> new_shape = {3, 4};
    deep_ros::Tensor reshaped = tensor.reshape(new_shape);

    REQUIRE(reshaped.shape() == new_shape);
    REQUIRE(reshaped.size() == tensor.size());
    REQUIRE(reshaped.data() == tensor.data());
    REQUIRE(reshaped.dtype() == tensor.dtype());
  }

  SECTION("Invalid reshape - size mismatch")
  {
    std::vector<size_t> invalid_shape = {2, 4};  // Size 8 vs 12
    REQUIRE_THROWS_AS(tensor.reshape(invalid_shape), std::invalid_argument);
  }
}

TEST_CASE("Data type sizes", "[tensor]")
{
  REQUIRE(get_dtype_size(deep_ros::DataType::FLOAT32) == sizeof(float));
  REQUIRE(get_dtype_size(deep_ros::DataType::FLOAT64) == sizeof(double));
  REQUIRE(get_dtype_size(deep_ros::DataType::INT8) == sizeof(int8_t));
  REQUIRE(get_dtype_size(deep_ros::DataType::INT16) == sizeof(int16_t));
  REQUIRE(get_dtype_size(deep_ros::DataType::INT32) == sizeof(int32_t));
  REQUIRE(get_dtype_size(deep_ros::DataType::INT64) == sizeof(int64_t));
  REQUIRE(get_dtype_size(deep_ros::DataType::UINT8) == sizeof(uint8_t));
  REQUIRE(get_dtype_size(deep_ros::DataType::UINT16) == sizeof(uint16_t));
  REQUIRE(get_dtype_size(deep_ros::DataType::UINT32) == sizeof(uint32_t));
  REQUIRE(get_dtype_size(deep_ros::DataType::UINT64) == sizeof(uint64_t));
  REQUIRE(get_dtype_size(deep_ros::DataType::BOOL) == sizeof(bool));
}

TEST_CASE("deep_ros::Tensor is_contiguous", "[tensor]")
{
  auto allocator = std::make_shared<deep_ros::test::MockMemoryAllocator>();
  std::vector<size_t> shape = {2, 3, 4};
  deep_ros::Tensor tensor(shape, deep_ros::DataType::FLOAT32, allocator);

  REQUIRE(tensor.is_contiguous() == true);
}

TEST_CASE("deep_ros::Tensor strides calculation", "[tensor]")
{
  auto allocator = std::make_shared<deep_ros::test::MockMemoryAllocator>();

  SECTION("2D tensor strides")
  {
    std::vector<size_t> shape = {3, 4};
    deep_ros::Tensor tensor(shape, deep_ros::DataType::FLOAT32, allocator);

    auto strides = tensor.strides();
    REQUIRE(strides.size() == 2);
    REQUIRE(strides[0] == 4 * sizeof(float));  // Row stride
    REQUIRE(strides[1] == sizeof(float));  // Column stride
  }

  SECTION("3D tensor strides")
  {
    std::vector<size_t> shape = {2, 3, 4};
    deep_ros::Tensor tensor(shape, deep_ros::DataType::INT16, allocator);

    auto strides = tensor.strides();
    REQUIRE(strides.size() == 3);
    REQUIRE(strides[0] == 3 * 4 * sizeof(int16_t));  // Depth stride
    REQUIRE(strides[1] == 4 * sizeof(int16_t));  // Row stride
    REQUIRE(strides[2] == sizeof(int16_t));  // Column stride
  }
}

TEST_CASE("deep_ros::Tensor assignment operators", "[tensor]")
{
  auto allocator = std::make_shared<deep_ros::test::MockMemoryAllocator>();
  std::vector<size_t> shape = {2, 2};
  deep_ros::Tensor original(shape, deep_ros::DataType::FLOAT32, allocator);

  // Fill with test data
  auto data = original.data_as<float>();
  data[0] = 1.0f;
  data[1] = 2.0f;
  data[2] = 3.0f;
  data[3] = 4.0f;

  SECTION("Copy assignment")
  {
    deep_ros::Tensor copy;
    copy = original;

    REQUIRE(copy.shape() == original.shape());
    REQUIRE(copy.data() != original.data());

    auto copy_data = copy.data_as<float>();
    REQUIRE(copy_data[0] == 1.0f);
    REQUIRE(copy_data[3] == 4.0f);
  }

  SECTION("Move assignment")
  {
    void * original_ptr = original.data();
    deep_ros::Tensor moved;
    moved = std::move(original);

    REQUIRE(moved.shape() == shape);
    REQUIRE(moved.data() == original_ptr);
    REQUIRE(original.data() == nullptr);
  }
}
