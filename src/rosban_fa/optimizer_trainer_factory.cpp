#include "rosban_fa/optimizer_trainer_factory.h"

#include "rosban_fa/adaptative_tree.h"

namespace rosban_fa
{

OptimizerTrainerFactory::OptimizerTrainerFactory()
{
  registerBuilder("AdaptativeTree",
                  []() { return std::unique_ptr<OptimizerTrainer>(new AdaptativeTree); });
}

}
