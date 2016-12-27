#include "rosban_fa/pwl_forest.h"

namespace rosban_fa
{

PWLForest::PWLForest() {}

PWLForest::PWLForest(std::unique_ptr<Forests> forests_,
                     int max_action_tiles_)
  : ForestApproximator(std::move(forests_), max_action_tiles_)
{}

PWLForest::~PWLForest() {}

std::unique_ptr<FunctionApproximator> PWLForest::clone() const {
  throw std::logic_error("PWLForest::clone: not implemented");
}

int PWLForest::getClassID() const
{
  return FunctionApproximator::PWLForest;
}

}
