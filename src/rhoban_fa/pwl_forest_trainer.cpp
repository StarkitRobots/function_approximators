#include "rhoban_fa/pwl_forest_trainer.h"

using regression_forests::Approximation;

namespace rhoban_fa
{

PWLForestTrainer::PWLForestTrainer() {}
PWLForestTrainer::~PWLForestTrainer() {}

regression_forests::Approximation::ID PWLForestTrainer::getApproximationID() const
{
  return Approximation::ID::PWL;
}

std::string PWLForestTrainer::getClassName() const
{
  return "PWLForestTrainer";
}

}
