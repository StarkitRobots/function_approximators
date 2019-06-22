#include "starkit_fa/optimizer_trainer_factory.h"

#include "starkit_fa/adaptative_tree.h"

namespace starkit_fa
{
OptimizerTrainerFactory::OptimizerTrainerFactory()
{
  registerBuilder("AdaptativeTree", []() { return std::unique_ptr<OptimizerTrainer>(new AdaptativeTree); });
}

}  // namespace starkit_fa
