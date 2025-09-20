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

#include "onnxruntime_gpu/onnx_cuda_allocator.hpp"

#include <memory>
#include <stdexcept>
#include <string>

#ifdef ONNX_GPU_CUDA_ENABLED
  #include <cuda_runtime.h>
#endif

namespace onnxruntime_gpu
{

#ifdef ONNX_GPU_CUDA_ENABLED

void OnnxCudaAllocator::check_cuda_error(int error, const std::string & operation)
{
  if (error != cudaSuccess) {
    throw std::runtime_error("CUDA error in " + operation + ": " + cudaGetErrorString(static_cast<cudaError_t>(error)));
  }
}

OnnxCudaAllocator::OnnxCudaAllocator(int device_id)
: device_id_(device_id)
{
  int device_count;
  check_cuda_error(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");

  if (device_id_ >= device_count) {
    throw std::invalid_argument("Invalid CUDA device ID: " + std::to_string(device_id_));
  }

  check_cuda_error(cudaSetDevice(device_id_), "cudaSetDevice");

  // TODO(wato): Initialize ONNX Runtime CUDA provider specific allocator
}

OnnxCudaAllocator::~OnnxCudaAllocator() = default;

void * OnnxCudaAllocator::allocate(size_t bytes)
{
  check_cuda_error(cudaSetDevice(device_id_), "cudaSetDevice");

  void * ptr = nullptr;
  check_cuda_error(cudaMalloc(&ptr, bytes), "cudaMalloc");
  return ptr;
}

void OnnxCudaAllocator::deallocate(void * ptr)
{
  if (ptr) {
    check_cuda_error(cudaSetDevice(device_id_), "cudaSetDevice");
    check_cuda_error(cudaFree(ptr), "cudaFree");
  }
}

void OnnxCudaAllocator::copy_from_host(void * dst, const void * src, size_t bytes)
{
  check_cuda_error(cudaSetDevice(device_id_), "cudaSetDevice");
  check_cuda_error(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
}

void OnnxCudaAllocator::copy_to_host(void * dst, const void * src, size_t bytes)
{
  check_cuda_error(cudaSetDevice(device_id_), "cudaSetDevice");
  check_cuda_error(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
}

void OnnxCudaAllocator::copy_device_to_device(void * dst, const void * src, size_t bytes)
{
  check_cuda_error(cudaSetDevice(device_id_), "cudaSetDevice");
  check_cuda_error(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice), "cudaMemcpy D2D");
}

bool OnnxCudaAllocator::is_device_memory() const
{
  return true;
}

std::string OnnxCudaAllocator::device_name() const
{
  return "onnx_cuda:" + std::to_string(device_id_);
}

bool OnnxCudaAllocator::is_cuda_available()
{
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  return error == cudaSuccess && device_count > 0;
}

std::shared_ptr<deep_ros::MemoryAllocator> get_onnx_cuda_allocator(int device_id)
{
  try {
    return std::make_shared<OnnxCudaAllocator>(device_id);
  } catch (const std::exception &) {
    return nullptr;
  }
}

#else  // ONNX_GPU_CUDA_ENABLED

OnnxCudaAllocator::OnnxCudaAllocator(int device_id)
: device_id_(device_id)
{
  throw std::runtime_error("CUDA support not compiled in this build");
}

OnnxCudaAllocator::~OnnxCudaAllocator() = default;

void * OnnxCudaAllocator::allocate(size_t /*bytes*/)  // NOLINT(readability/casting)
{
  throw std::runtime_error("CUDA support not compiled in this build");
}

void OnnxCudaAllocator::deallocate(void * /*ptr*/)  // NOLINT(readability/casting)
{
  throw std::runtime_error("CUDA support not compiled in this build");
}

void OnnxCudaAllocator::copy_from_host(void *, const void *, size_t)
{
  throw std::runtime_error("CUDA support not compiled in this build");
}

void OnnxCudaAllocator::copy_to_host(void *, const void *, size_t)
{
  throw std::runtime_error("CUDA support not compiled in this build");
}

void OnnxCudaAllocator::copy_device_to_device(void *, const void *, size_t)
{
  throw std::runtime_error("CUDA support not compiled in this build");
}

bool OnnxCudaAllocator::is_device_memory() const
{
  return true;
}

std::string OnnxCudaAllocator::device_name() const
{
  return "onnx_cuda:" + std::to_string(device_id_) + " (no_cuda)";
}

bool OnnxCudaAllocator::is_cuda_available()
{
  return false;
}

std::shared_ptr<deep_ros::MemoryAllocator> get_onnx_cuda_allocator(int /*device_id*/)  // NOLINT(readability/casting)
{
  return nullptr;
}

#endif  // ONNX_GPU_CUDA_ENABLED

}  // namespace onnxruntime_gpu
