#pragma once

#include "rhoban_fa/forest_trainer.h"

namespace rhoban_fa
{
class PWLForestTrainer : public ForestTrainer
{
public:
  PWLForestTrainer();
  ~PWLForestTrainer();

  virtual regression_forests::Approximation::ID getApproximationID() const override;

  virtual std::string getClassName() const override;
};

}  // namespace rhoban_fa
