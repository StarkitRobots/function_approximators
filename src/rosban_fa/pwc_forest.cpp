#include "rosban_fa/pwc_forest.h"

namespace rosban_fa
{

PWCForest::PWCForest() {}

PWCForest::PWCForest(std::unique_ptr<Forests> forests_,
                     int max_action_tiles_)
  : ForestApproximator(std::move(forests_), max_action_tiles_)
{}

PWCForest::~PWCForest() {}

std::unique_ptr<FunctionApproximator> PWCForest::clone() const {
  throw std::logic_error("PWCForest::clone: not implemented");
}

int PWCForest::getClassID() const
{
  return FunctionApproximator::PWCForest;
}


}
