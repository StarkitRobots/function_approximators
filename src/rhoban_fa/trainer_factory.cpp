#include "rhoban_fa/trainer_factory.h"

#include "rhoban_fa/pwc_forest_trainer.h"
#include "rhoban_fa/pwl_forest_trainer.h"

#ifdef RHOBAN_FA_USES_DNN
#include "rhoban_fa/dnn_approximator_trainer.h"
#endif
#ifdef RHOBAN_FA_USES_GP
#include "rhoban_fa/gp_trainer.h"
#include "rhoban_fa/gp_forest_trainer.h"
#endif

namespace rhoban_fa
{

TrainerFactory::TrainerFactory()
{
  registerBuilder("PWCForestTrainer", [](){return std::unique_ptr<Trainer>(new PWCForestTrainer);});
  registerBuilder("PWLForestTrainer", [](){return std::unique_ptr<Trainer>(new PWLForestTrainer);});
#ifdef RHOBAN_FA_USES_DNN
  registerBuilder("DNNApproximatorTrainer",
                  [](){return std::unique_ptr<Trainer>(new DNNApproximatorTrainer);});
#endif
#ifdef RHOBAN_FA_USES_GP
  registerBuilder("GPTrainer"       , [](){return std::unique_ptr<Trainer>(new GPTrainer);       });
  registerBuilder("GPForestTrainer" , [](){return std::unique_ptr<Trainer>(new GPForestTrainer); });
#endif
}

}
