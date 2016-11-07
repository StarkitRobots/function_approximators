#pragma once

#include "rosban_fa/optimizer_trainer.h"

#include "rosban_utils/factory.h"

#include <map>

namespace rosban_fa
{

class OptimizerTrainerFactory : public rosban_utils::Factory<OptimizerTrainer>
{
public:
  OptimizerTrainerFactory();
};

}
