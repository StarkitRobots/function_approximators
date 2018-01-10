#pragma once

#include "rosban_fa/function_approximator.h"

#include "rhoban_utils/serialization/factory.h"

#include <map>

namespace rosban_fa
{

class FunctionApproximatorFactory : public rhoban_utils::Factory<FunctionApproximator>
{
public:
  FunctionApproximatorFactory();
};

}
