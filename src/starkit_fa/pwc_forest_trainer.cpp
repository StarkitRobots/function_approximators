#include "starkit_fa/pwc_forest_trainer.h"

using regression_forests::Approximation;

namespace starkit_fa
{
PWCForestTrainer::PWCForestTrainer()
{
}
PWCForestTrainer::~PWCForestTrainer()
{
}

regression_forests::Approximation::ID PWCForestTrainer::getApproximationID() const
{
  return Approximation::ID::PWC;
}

std::string PWCForestTrainer::getClassName() const
{
  return "PWCForestTrainer";
}

}  // namespace starkit_fa
