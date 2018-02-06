#pragma once

#include "rhoban_fa/split.h"

#include "rhoban_utils/serialization/factory.h"

#include <map>

namespace rhoban_fa
{

class SplitFactory : public rhoban_utils::Factory<Split>
{
public:
  SplitFactory();
};

}
