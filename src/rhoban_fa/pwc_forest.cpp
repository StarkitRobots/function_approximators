#include "rhoban_fa/pwc_forest.h"

namespace rhoban_fa
{

PWCForest::PWCForest() {}

PWCForest::PWCForest(std::unique_ptr<Forests> forests_,
                     int max_action_tiles_)
  : ForestApproximator(std::move(forests_), max_action_tiles_)
{}

PWCForest::~PWCForest() {}

std::unique_ptr<FunctionApproximator> PWCForest::clone() const {
  std::unique_ptr<PWCForest> copy(new PWCForest);
  copy->forests = cloneForests(*forests);
  copy->max_action_tiles = max_action_tiles;
  copy->aggregation_method = aggregation_method;
  return std::move(copy);
}

int PWCForest::getClassID() const
{
  return FunctionApproximator::PWCForest;
}


}
