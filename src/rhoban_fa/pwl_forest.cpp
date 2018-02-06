#include "rhoban_fa/pwl_forest.h"

namespace rhoban_fa
{

PWLForest::PWLForest() {}

PWLForest::PWLForest(std::unique_ptr<Forests> forests_,
                     int max_action_tiles_)
  : ForestApproximator(std::move(forests_), max_action_tiles_)
{}

PWLForest::~PWLForest() {}

std::unique_ptr<FunctionApproximator> PWLForest::clone() const {
  std::unique_ptr<PWLForest> copy(new PWLForest);
  copy->forests = cloneForests(*forests);
  copy->max_action_tiles = max_action_tiles;
  copy->aggregation_method = aggregation_method;
  return std::move(copy);
}

int PWLForest::getClassID() const
{
  return FunctionApproximator::PWLForest;
}

}
