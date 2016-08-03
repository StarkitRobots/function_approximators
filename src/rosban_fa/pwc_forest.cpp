#include "rosban_fa/pwc_forest.h"

namespace rosban_fa
{

PWCForest::PWCForest() {}

PWCForest::PWCForest(std::unique_ptr<Forests> forests_,
                     int max_action_tiles_)
  : ForestApproximator(std::move(forests_), max_action_tiles_)
{}

PWCForest::~PWCForest() {}

int PWCForest::getClassID() const
{
  return FunctionApproximator::PWCForest;
}


}
