#pragma once

#include "rosban_fa/optimizer_trainer.h"

#include "rhoban_utils/serialization/factory.h"

#include <map>

namespace rosban_fa
{

class OptimizerTrainerFactory : public rhoban_utils::Factory<OptimizerTrainer>
{
public:
  OptimizerTrainerFactory();
};

}
