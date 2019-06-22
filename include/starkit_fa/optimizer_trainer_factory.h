#pragma once

#include "starkit_fa/optimizer_trainer.h"

#include "starkit_utils/serialization/factory.h"

#include <map>

namespace starkit_fa
{
class OptimizerTrainerFactory : public starkit_utils::Factory<OptimizerTrainer>
{
public:
  OptimizerTrainerFactory();
};

}  // namespace starkit_fa
