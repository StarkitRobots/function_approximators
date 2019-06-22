#include "starkit_fa/split_factory.h"

#include "starkit_fa/fake_split.h"
#include "starkit_fa/linear_split.h"
#include "starkit_fa/orthogonal_split.h"
#include "starkit_fa/point_split.h"

namespace starkit_fa
{
SplitFactory::SplitFactory()
{
  registerBuilder(Split::Orthogonal, []() { return std::unique_ptr<Split>(new OrthogonalSplit); });
  registerBuilder(Split::Point, []() { return std::unique_ptr<Split>(new PointSplit); });
  registerBuilder(Split::Fake, []() { return std::unique_ptr<Split>(new FakeSplit); });
  registerBuilder(Split::Linear, []() { return std::unique_ptr<Split>(new LinearSplit); });
}

}  // namespace starkit_fa
