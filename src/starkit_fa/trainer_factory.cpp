#include "starkit_fa/trainer_factory.h"

#include "starkit_fa/pwc_forest_trainer.h"
#include "starkit_fa/pwl_forest_trainer.h"

#ifdef STARKIT_FA_USES_DNN
#include "starkit_fa/dnn_approximator_trainer.h"
#endif
#ifdef STARKIT_FA_USES_GP
#include "starkit_fa/gp_trainer.h"
#include "starkit_fa/gp_forest_trainer.h"
#endif

namespace starkit_fa
{
TrainerFactory::TrainerFactory()
{
  registerBuilder("PWCForestTrainer", []() { return std::unique_ptr<Trainer>(new PWCForestTrainer); });
  registerBuilder("PWLForestTrainer", []() { return std::unique_ptr<Trainer>(new PWLForestTrainer); });
#ifdef STARKIT_FA_USES_DNN
  registerBuilder("DNNApproximatorTrainer", []() { return std::unique_ptr<Trainer>(new DNNApproximatorTrainer); });
#endif
#ifdef STARKIT_FA_USES_GP
  registerBuilder("GPTrainer", []() { return std::unique_ptr<Trainer>(new GPTrainer); });
  registerBuilder("GPForestTrainer", []() { return std::unique_ptr<Trainer>(new GPForestTrainer); });
#endif
}

}  // namespace starkit_fa
