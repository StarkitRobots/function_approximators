#include "rosban_fa/pwl_forest_trainer.h"

using regression_forests::Approximation;

namespace rosban_fa
{

PWLForestTrainer::PWLForestTrainer() {}
PWLForestTrainer::~PWLForestTrainer() {}

regression_forests::Approximation::ID PWLForestTrainer::getApproximationID() const
{
  return Approximation::ID::PWL;
}

std::string PWLForestTrainer::class_name() const
{
  return "PWLForestTrainer";
}

}
