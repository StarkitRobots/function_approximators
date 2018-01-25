#include "rosban_fa/trainer_factory.h"

#include "rosban_fa/dnn_approximator_trainer.h"
#include "rosban_fa/gp_trainer.h"
#include "rosban_fa/gp_forest_trainer.h"
#include "rosban_fa/pwc_forest_trainer.h"
#include "rosban_fa/pwl_forest_trainer.h"

namespace rosban_fa
{

TrainerFactory::TrainerFactory()
{
  registerBuilder("DNNApproximatorTrainer"       ,
                  [](){return std::unique_ptr<Trainer>(new DNNApproximatorTrainer);});
  registerBuilder("GPTrainer"       , [](){return std::unique_ptr<Trainer>(new GPTrainer);       });
  registerBuilder("GPForestTrainer" , [](){return std::unique_ptr<Trainer>(new GPForestTrainer); });
  registerBuilder("PWCForestTrainer", [](){return std::unique_ptr<Trainer>(new PWCForestTrainer);});
  registerBuilder("PWLForestTrainer", [](){return std::unique_ptr<Trainer>(new PWLForestTrainer);});
}

}
