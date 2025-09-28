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

#include "deep_ort_backend_plugin/ort_cpu_memory_allocator.hpp"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace deep_ort_backend
{

// Static instance pointer for callbacks
OrtCpuMemoryAllocator * OrtCpuMemoryAllocator::instance_ = nullptr;

OrtCpuMemoryAllocator::OrtCpuMemoryAllocator()
: ort_memory_info_(nullptr)
{
  // Set the static instance pointer for callbacks
  instance_ = this;

  // Initialize OrtAllocator struct
  ort_allocator_.version = ORT_API_VERSION;
  ort_allocator_.Alloc = &OrtCpuMemoryAllocator::ort_alloc;
  ort_allocator_.Free = &OrtCpuMemoryAllocator::ort_free;
  ort_allocator_.Info = &OrtCpuMemoryAllocator::ort_info;
  ort_allocator_.Reserve = &OrtCpuMemoryAllocator::ort_reserve;

  // Create OrtMemoryInfo for CPU allocator
  OrtStatus * status = OrtGetApiBase()
                         ->GetApi(ORT_API_VERSION)
                         ->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ort_memory_info_);
  if (status != nullptr) {
    OrtGetApiBase()->GetApi(ORT_API_VERSION)->ReleaseStatus(status);
    throw std::runtime_error("Failed to create OrtMemoryInfo");
  }
}

OrtCpuMemoryAllocator::~OrtCpuMemoryAllocator()
{
  if (ort_memory_info_) {
    OrtGetApiBase()->GetApi(ORT_API_VERSION)->ReleaseMemoryInfo(ort_memory_info_);
  }
}

OrtAllocator * OrtCpuMemoryAllocator::get_ort_allocator()
{
  return &ort_allocator_;
}

const OrtMemoryInfo * OrtCpuMemoryAllocator::get_ort_memory_info() const
{
  return ort_memory_info_;
}

void * ORT_API_CALL OrtCpuMemoryAllocator::ort_alloc(OrtAllocator * this_, size_t size)
{
  if (instance_) {
    return instance_->allocate(size);
  }
  return nullptr;
}

void ORT_API_CALL OrtCpuMemoryAllocator::ort_free(OrtAllocator * this_, void * p)
{
  if (instance_) {
    instance_->deallocate(p);
  }
}

const OrtMemoryInfo * ORT_API_CALL OrtCpuMemoryAllocator::ort_info(const OrtAllocator * this_)
{
  if (instance_) {
    return instance_->get_ort_memory_info();
  }
  return nullptr;
}

void * ORT_API_CALL OrtCpuMemoryAllocator::ort_reserve(OrtAllocator * this_, size_t size)
{
  // For CPU allocator, Reserve can be the same as Alloc
  if (instance_) {
    return instance_->allocate(size);
  }
  return nullptr;
}

void * OrtCpuMemoryAllocator::allocate(size_t bytes)
{
  if (bytes == 0) {
    return nullptr;
  }

  // Use 64-byte alignment for optimal SIMD performance (AVX-512)
  void * ptr = std::aligned_alloc(64, bytes);
  if (!ptr) {
    throw std::bad_alloc();
  }
  return ptr;
}

void OrtCpuMemoryAllocator::deallocate(void * ptr)
{
  if (ptr) {
    std::free(ptr);
  }
}

void OrtCpuMemoryAllocator::copy_from_host_impl(void * dst, const void * src, size_t bytes)
{
  if (dst && src && bytes > 0) {
    std::memcpy(dst, src, bytes);
  }
}

void OrtCpuMemoryAllocator::copy_to_host_impl(void * dst, const void * src, size_t bytes)
{
  if (dst && src && bytes > 0) {
    std::memcpy(dst, src, bytes);
  }
}

void OrtCpuMemoryAllocator::copy_device_to_device_impl(void * dst, const void * src, size_t bytes)
{
  if (dst && src && bytes > 0) {
    std::memcpy(dst, src, bytes);
  }
}

bool OrtCpuMemoryAllocator::is_device_memory() const
{
  return false;  // CPU memory is host memory
}

std::string OrtCpuMemoryAllocator::device_name() const
{
  return "ort_cpu";
}

std::shared_ptr<deep_ros::BackendMemoryAllocator> get_ort_cpu_allocator()
{
  static std::shared_ptr<deep_ros::BackendMemoryAllocator> allocator = std::make_shared<OrtCpuMemoryAllocator>();
  return allocator;
}

}  // namespace deep_ort_backend
