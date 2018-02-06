#pragma once

#include "rhoban_fa/optimizer_trainer.h"

#include "rhoban_utils/serialization/factory.h"

#include <map>

namespace rhoban_fa
{

class OptimizerTrainerFactory : public rhoban_utils::Factory<OptimizerTrainer>
{
public:
  OptimizerTrainerFactory();
};

}
