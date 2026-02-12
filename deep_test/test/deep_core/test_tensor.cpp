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
#include <memory>
#include <utility>
#include <vector>

#include <deep_core/types/tensor.hpp>
#include <deep_test/compat.hpp>
#include <test_fixtures/mock_backend_fixture.hpp>

TEST_CASE("deep_ros::Tensor: Basic Construction and Properties", "[tensor][construction]")
{
  SECTION("Default constructor creates empty tensor")
  {
    deep_ros::Tensor tensor;

    REQUIRE(tensor.rank() == 0);
    REQUIRE(tensor.size() == 0);
    REQUIRE(tensor.byte_size() == 0);
    REQUIRE(tensor.data() == nullptr);
    REQUIRE(tensor.shape().empty());
    REQUIRE(tensor.dtype() == deep_ros::DataType::FLOAT32);  // Should have sensible default
  }

  SECTION("External data constructor doesn't take ownership")
  {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<size_t> shape = {2, 2};

    deep_ros::Tensor tensor(data.data(), shape, deep_ros::DataType::FLOAT32);

    // Should point to the same memory
    REQUIRE(tensor.data() == data.data());
    REQUIRE(tensor.shape() == shape);
    REQUIRE(tensor.size() == 4);
    REQUIRE(tensor.dtype() == deep_ros::DataType::FLOAT32);
    REQUIRE(tensor.byte_size() == 4 * sizeof(float));

    // Modifying original should affect tensor
    data[0] = 999.0f;
    auto tensor_data = tensor.data_as<float>();
    REQUIRE(tensor_data[0] == 999.0f);
  }
}

TEST_CASE_METHOD(
  deep_ros::test::MockBackendFixture, "deep_ros::Tensor: Managed Memory Construction", "[tensor][memory]")
{
  auto allocator = getAllocator();

  SECTION("Allocator-based constructor takes ownership")
  {
    std::vector<size_t> shape = {3, 4};
    deep_ros::Tensor tensor(shape, deep_ros::DataType::INT32, allocator);

    REQUIRE(tensor.shape() == shape);
    REQUIRE(tensor.size() == 12);
    REQUIRE(tensor.byte_size() == 12 * sizeof(int32_t));
    REQUIRE(tensor.data() != nullptr);
    REQUIRE(tensor.dtype() == deep_ros::DataType::INT32);

    // Should be able to write to memory
    auto data = tensor.data_as<int32_t>();
    for (size_t i = 0; i < tensor.size(); ++i) {
      data[i] = static_cast<int32_t>(i * 10);
    }

    // Verify data persistence
    for (size_t i = 0; i < tensor.size(); ++i) {
      REQUIRE(data[i] == static_cast<int32_t>(i * 10));
    }
  }

  SECTION("Different data types work correctly")
  {
    std::vector<size_t> shape = {2, 3};

    deep_ros::Tensor float32_tensor(shape, deep_ros::DataType::FLOAT32, allocator);
    deep_ros::Tensor float64_tensor(shape, deep_ros::DataType::FLOAT64, allocator);
    deep_ros::Tensor uint8_tensor(shape, deep_ros::DataType::UINT8, allocator);
    deep_ros::Tensor bool_tensor(shape, deep_ros::DataType::BOOL, allocator);

    REQUIRE(float32_tensor.byte_size() == 6 * 4);
    REQUIRE(float64_tensor.byte_size() == 6 * 8);
    REQUIRE(uint8_tensor.byte_size() == 6 * 1);
    REQUIRE(bool_tensor.byte_size() == 6 * 1);
  }
}

TEST_CASE("deep_ros::Tensor: Input Validation", "[tensor][validation]")
{
  std::vector<float> data = {1.0f, 2.0f};

  SECTION("Empty shape should be rejected")
  {
    std::vector<size_t> empty_shape;
    REQUIRE_THROWS_AS(deep_ros::Tensor(data.data(), empty_shape, deep_ros::DataType::FLOAT32), std::invalid_argument);
  }

  SECTION("Zero dimensions should be rejected")
  {
    std::vector<size_t> zero_shape = {2, 0, 3};
    REQUIRE_THROWS_AS(deep_ros::Tensor(data.data(), zero_shape, deep_ros::DataType::FLOAT32), std::invalid_argument);
  }

  SECTION("Null data pointer should be handled gracefully")
  {
    std::vector<size_t> shape = {2, 2};
    // This should either throw or create a valid tensor with null data
    // The behavior should be well-defined
    REQUIRE_NOTHROW(deep_ros::Tensor(nullptr, shape, deep_ros::DataType::FLOAT32));
  }
}

TEST_CASE_METHOD(deep_ros::test::MockBackendFixture, "deep_ros::Tensor: Memory Management", "[tensor][memory]")
{
  auto allocator = getAllocator();

  SECTION("Copy constructor creates independent copy")
  {
    std::vector<size_t> shape = {2, 3};
    deep_ros::Tensor original(shape, deep_ros::DataType::FLOAT32, allocator);

    // Fill with test data
    auto orig_data = original.data_as<float>();
    for (size_t i = 0; i < original.size(); ++i) {
      orig_data[i] = static_cast<float>(i + 1);
    }

    deep_ros::Tensor copy(original);

    // Should have same properties but different memory
    REQUIRE(copy.shape() == original.shape());
    REQUIRE(copy.dtype() == original.dtype());
    REQUIRE(copy.size() == original.size());
    REQUIRE(copy.data() != original.data());  // Different memory

    // Data should be identical
    auto copy_data = copy.data_as<float>();
    for (size_t i = 0; i < copy.size(); ++i) {
      REQUIRE(copy_data[i] == orig_data[i]);
    }

    // Modifying copy shouldn't affect original
    copy_data[0] = 999.0f;
    REQUIRE(orig_data[0] != copy_data[0]);
  }

  SECTION("Move constructor transfers ownership")
  {
    std::vector<size_t> shape = {3, 3};
    deep_ros::Tensor original(shape, deep_ros::DataType::UINT8, allocator);
    void * original_ptr = original.data();

    deep_ros::Tensor moved(std::move(original));

    // Moved tensor should have the data
    REQUIRE(moved.data() == original_ptr);
    REQUIRE(moved.size() == 9);
    REQUIRE(moved.shape() == shape);

    // Original should be empty
    REQUIRE(original.data() == nullptr);
    REQUIRE(original.size() == 0);
  }

  SECTION("Assignment operators work correctly")
  {
    std::vector<size_t> shape = {2, 2};
    deep_ros::Tensor source(shape, deep_ros::DataType::INT32, allocator);
    auto source_data = source.data_as<int32_t>();
    source_data[0] = 42;

    deep_ros::Tensor target;
    target = source;  // Copy assignment

    REQUIRE(target.size() == source.size());
    REQUIRE(target.data() != source.data());
    REQUIRE(target.data_as<int32_t>()[0] == 42);
  }
}

TEST_CASE_METHOD(deep_ros::test::MockBackendFixture, "deep_ros::Tensor: Shape Operations", "[tensor][shape]")
{
  auto allocator = getAllocator();

  SECTION("Reshape preserves data but changes shape")
  {
    std::vector<size_t> shape = {2, 3, 4};  // 24 elements
    deep_ros::Tensor tensor(shape, deep_ros::DataType::FLOAT32, allocator);

    // Fill with test pattern
    auto data = tensor.data_as<float>();
    for (size_t i = 0; i < tensor.size(); ++i) {
      data[i] = static_cast<float>(i);
    }

    // Valid reshapes
    auto reshaped1 = tensor.reshape({24});
    auto reshaped2 = tensor.reshape({6, 4});
    auto reshaped3 = tensor.reshape({4, 3, 2});

    REQUIRE(reshaped1.size() == 24);
    REQUIRE(reshaped2.size() == 24);
    REQUIRE(reshaped3.size() == 24);

    // Data should be preserved
    REQUIRE(reshaped1.data_as<float>()[0] == 0.0f);
    REQUIRE(reshaped1.data_as<float>()[23] == 23.0f);
  }

  SECTION("Invalid reshape should be rejected")
  {
    std::vector<size_t> shape = {2, 3, 4};  // 24 elements
    deep_ros::Tensor tensor(shape, deep_ros::DataType::FLOAT32, allocator);

    // Wrong total size
    REQUIRE_THROWS_AS(tensor.reshape({25}), std::invalid_argument);
    REQUIRE_THROWS_AS(tensor.reshape({2, 3, 5}), std::invalid_argument);

    // Zero dimensions
    REQUIRE_THROWS_AS(tensor.reshape({0, 24}), std::invalid_argument);
  }

  SECTION("deep_ros::Tensor properties are calculated correctly")
  {
    std::vector<size_t> shape = {2, 3, 4};
    deep_ros::Tensor tensor(shape, deep_ros::DataType::FLOAT64, allocator);

    REQUIRE(tensor.rank() == 3);
    REQUIRE(tensor.size() == 24);
    REQUIRE(tensor.byte_size() == 24 * sizeof(double));
    REQUIRE(tensor.is_contiguous() == true);

    // Strides should be calculated correctly
    auto strides = tensor.strides();
    REQUIRE(strides.size() == 3);
    REQUIRE(strides[2] == sizeof(double));  // Innermost stride
    REQUIRE(strides[1] == 4 * sizeof(double));  // Next stride
    REQUIRE(strides[0] == 12 * sizeof(double));  // Outermost stride
  }
}

TEST_CASE("deep_ros::Tensor: Data Type System", "[tensor][types]")
{
  SECTION("Data type sizes are correct")
  {
    REQUIRE(get_dtype_size(deep_ros::DataType::FLOAT32) == 4);
    REQUIRE(get_dtype_size(deep_ros::DataType::FLOAT64) == 8);
    REQUIRE(get_dtype_size(deep_ros::DataType::INT8) == 1);
    REQUIRE(get_dtype_size(deep_ros::DataType::INT16) == 2);
    REQUIRE(get_dtype_size(deep_ros::DataType::INT32) == 4);
    REQUIRE(get_dtype_size(deep_ros::DataType::INT64) == 8);
    REQUIRE(get_dtype_size(deep_ros::DataType::UINT8) == 1);
    REQUIRE(get_dtype_size(deep_ros::DataType::UINT16) == 2);
    REQUIRE(get_dtype_size(deep_ros::DataType::UINT32) == 4);
    REQUIRE(get_dtype_size(deep_ros::DataType::UINT64) == 8);
    REQUIRE(get_dtype_size(deep_ros::DataType::BOOL) == 1);
  }

  SECTION("Type casting works safely")
  {
    std::vector<float> data = {1.5f, 2.5f, 3.5f, 4.5f};
    std::vector<size_t> shape = {2, 2};

    deep_ros::Tensor tensor(data.data(), shape, deep_ros::DataType::FLOAT32);

    auto float_ptr = tensor.data_as<float>();
    REQUIRE(float_ptr != nullptr);
    REQUIRE(float_ptr[0] == 1.5f);
    REQUIRE(float_ptr[3] == 4.5f);

    // Should be same as raw data
    REQUIRE(float_ptr == tensor.data());
  }
}

TEST_CASE_METHOD(deep_ros::test::MockBackendFixture, "deep_ros::Tensor: Edge Cases", "[tensor][edge_cases]")
{
  auto allocator = getAllocator();

  SECTION("Single element tensor")
  {
    std::vector<size_t> shape = {1};
    deep_ros::Tensor tensor(shape, deep_ros::DataType::FLOAT32, allocator);

    REQUIRE(tensor.size() == 1);
    REQUIRE(tensor.rank() == 1);

    auto data = tensor.data_as<float>();
    data[0] = 42.0f;
    REQUIRE(data[0] == 42.0f);
  }

  SECTION("High-dimensional tensor")
  {
    std::vector<size_t> shape = {2, 2, 2, 2, 2};  // 5D tensor
    deep_ros::Tensor tensor(shape, deep_ros::DataType::UINT8, allocator);

    REQUIRE(tensor.rank() == 5);
    REQUIRE(tensor.size() == 32);  // 2^5
    REQUIRE(tensor.byte_size() == 32);
  }

  SECTION("Large tensor")
  {
    std::vector<size_t> shape = {1000, 1000};  // 1M elements
    deep_ros::Tensor tensor(shape, deep_ros::DataType::UINT8, allocator);

    REQUIRE(tensor.size() == 1000000);
    REQUIRE(tensor.byte_size() == 1000000);
    REQUIRE(tensor.data() != nullptr);
  }
}
