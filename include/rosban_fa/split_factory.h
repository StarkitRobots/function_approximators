#pragma once

#include "rosban_fa/split.h"

#include "rhoban_utils/serialization/factory.h"

#include <map>

namespace rosban_fa
{

class SplitFactory : public rhoban_utils::Factory<Split>
{
public:
  SplitFactory();
};

}
