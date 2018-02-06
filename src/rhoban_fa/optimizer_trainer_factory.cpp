#include "rhoban_fa/optimizer_trainer_factory.h"

#include "rhoban_fa/adaptative_tree.h"

namespace rhoban_fa
{

OptimizerTrainerFactory::OptimizerTrainerFactory()
{
  registerBuilder("AdaptativeTree",
                  []() { return std::unique_ptr<OptimizerTrainer>(new AdaptativeTree); });
}

}
