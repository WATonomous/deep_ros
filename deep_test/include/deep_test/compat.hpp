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

#ifndef DEEP_TEST__COMPAT_HPP_
  #define DEEP_TEST__COMPAT_HPP_

/**
 * Compatibility header for handling differences between ROS2 distributions
 * (primarily Humble vs Jazzy) and their respective dependencies.
 *
 * This header includes compatibility macros and includes for test code.
 */

// ============================================================================
// Catch2 Compatibility (2.x in Humble, 3.x in Jazzy+)
// ============================================================================

  #if __has_include(<catch2/catch_all.hpp>)
    #include <catch2/catch_all.hpp>
  #else
    #include <catch2/catch.hpp>
  #endif

  // Compatibility macro for Catch2 2.x vs 3.x Approx API
  // Catch2 2.x (Humble): Approx() in global namespace
  // Catch2 3.x (Jazzy+): Catch::Approx() in Catch namespace
  #if __has_include(<catch2/catch_all.hpp>)
    #define CATCH_APPROX(x) Catch::Approx(x)
  #else
    #define CATCH_APPROX(x) Approx(x)
  #endif

#endif  // DEEP_TEST__COMPAT_HPP_
