#pragma once

#include "rosban_fa/forest_trainer.h"

namespace rosban_fa
{

class PWCForestTrainer : public ForestTrainer
{
public:
  PWCForestTrainer();
  ~PWCForestTrainer();

  virtual regression_forests::Approximation::ID getApproximationID() const override;

  virtual std::string getClassName() const override;
};

}
