#pragma once

#include "rhoban_fa/function_approximator.h"

#include "rhoban_utils/serialization/factory.h"

#include <map>

namespace rhoban_fa
{
class FunctionApproximatorFactory : public rhoban_utils::Factory<FunctionApproximator>
{
public:
  FunctionApproximatorFactory();
};

}  // namespace rhoban_fa
