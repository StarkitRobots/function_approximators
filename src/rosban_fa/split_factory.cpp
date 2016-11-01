#include "rosban_fa/split_factory.h"

#include "rosban_fa/orthogonal_split.h"

namespace rosban_fa
{

SplitFactory::SplitFactory()
{
  registerBuilder(Split::Orthogonal,
                  []() { return std::unique_ptr<Split>(new OrthogonalSplit); });
}

}
