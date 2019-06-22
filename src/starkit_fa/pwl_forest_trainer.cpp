#include "starkit_fa/pwl_forest_trainer.h"

using regression_forests::Approximation;

namespace starkit_fa
{
PWLForestTrainer::PWLForestTrainer()
{
}
PWLForestTrainer::~PWLForestTrainer()
{
}

regression_forests::Approximation::ID PWLForestTrainer::getApproximationID() const
{
  return Approximation::ID::PWL;
}

std::string PWLForestTrainer::getClassName() const
{
  return "PWLForestTrainer";
}

}  // namespace starkit_fa
