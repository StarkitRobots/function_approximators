#include "rhoban_fa/split_factory.h"

#include "rhoban_fa/fake_split.h"
#include "rhoban_fa/linear_split.h"
#include "rhoban_fa/orthogonal_split.h"
#include "rhoban_fa/point_split.h"

namespace rhoban_fa
{
SplitFactory::SplitFactory()
{
  registerBuilder(Split::Orthogonal, []() { return std::unique_ptr<Split>(new OrthogonalSplit); });
  registerBuilder(Split::Point, []() { return std::unique_ptr<Split>(new PointSplit); });
  registerBuilder(Split::Fake, []() { return std::unique_ptr<Split>(new FakeSplit); });
  registerBuilder(Split::Linear, []() { return std::unique_ptr<Split>(new LinearSplit); });
}

}  // namespace rhoban_fa
