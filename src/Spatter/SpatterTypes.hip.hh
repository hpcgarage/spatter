/*!
  \file SpatterTypes.hip.hh
*/

#ifndef SPATTER_TYPES_HPP
#define SPATTER_TYPES_HPP

#include <type_traits>

namespace Spatter {
struct Serial {};
struct OpenMP {};
struct CUDA {};
struct HIP {};
} // namespace Spatter

#endif
