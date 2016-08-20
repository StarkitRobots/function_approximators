#pragma once

#include "rosban_fa/forest_trainer.h"

namespace rosban_fa
{

class PWLForestTrainer : public ForestTrainer
{
public:
  PWLForestTrainer();
  ~PWLForestTrainer();

  virtual regression_forests::Approximation::ID getApproximationID() const override;

  virtual std::string class_name() const override;
};

}
