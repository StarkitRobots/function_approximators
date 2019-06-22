#pragma once

#include "starkit_fa/forest_approximator.h"

namespace starkit_fa
{
class PWCForest : public ForestApproximator
{
public:
  PWCForest();
  PWCForest(std::unique_ptr<Forests> forests, int max_action_tiles);

  virtual ~PWCForest();

  virtual std::unique_ptr<FunctionApproximator> clone() const override;

  virtual int getClassID() const override;
};

}  // namespace starkit_fa
