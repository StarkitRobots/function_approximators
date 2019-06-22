#pragma once

#include "starkit_fa/split.h"

#include "starkit_utils/serialization/factory.h"

#include <map>

namespace starkit_fa
{
class SplitFactory : public starkit_utils::Factory<Split>
{
public:
  SplitFactory();
};

}  // namespace starkit_fa
