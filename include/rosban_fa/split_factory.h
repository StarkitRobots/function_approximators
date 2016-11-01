#pragma once

#include "rosban_fa/split.h"

#include "rosban_utils/factory.h"

#include <map>

namespace rosban_fa
{

class SplitFactory : public rosban_utils::Factory<Split>
{
public:
  SplitFactory();
};

}
