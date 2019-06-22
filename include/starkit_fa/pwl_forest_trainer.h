#pragma once

#include "starkit_fa/forest_trainer.h"

namespace starkit_fa
{
class PWLForestTrainer : public ForestTrainer
{
public:
  PWLForestTrainer();
  ~PWLForestTrainer();

  virtual regression_forests::Approximation::ID getApproximationID() const override;

  virtual std::string getClassName() const override;
};

}  // namespace starkit_fa
