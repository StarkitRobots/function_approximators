#pragma once

#include "rosban_fa/function_approximator.h"

#include "rosban_utils/factory.h"

#include <map>

namespace rosban_fa
{

class FunctionApproximatorFactory : public rosban_utils::Factory<FunctionApproximator>
{
public:
  FunctionApproximatorFactory();
};

}
