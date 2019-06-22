#pragma once

#include "starkit_fa/function_approximator.h"

#include "starkit_utils/serialization/factory.h"

#include <map>

namespace starkit_fa
{
class FunctionApproximatorFactory : public starkit_utils::Factory<FunctionApproximator>
{
public:
  FunctionApproximatorFactory();
};

}  // namespace starkit_fa
