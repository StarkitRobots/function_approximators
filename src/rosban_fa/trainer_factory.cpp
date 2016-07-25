#include "rosban_fa/trainer_factory.h"

#include "rosban_fa/gp_trainer.h"
#include "rosban_fa/gp_forest_trainer.h"
#include "rosban_fa/pwc_forest_trainer.h"
#include "rosban_fa/pwl_forest_trainer.h"

namespace rosban_fa
{

TrainerFactory::TrainerFactory()
{
  registerBuilder("GPTrainer",
                  [](TiXmlNode * node)
                  {
                    Trainer * trainer = new GPTrainer();
                    trainer->from_xml(node);
                    return trainer;
                  });
  registerBuilder("GPForestTrainer",
                  [](TiXmlNode * node)
                  {
                    Trainer * trainer = new GPForestTrainer();
                    trainer->from_xml(node);
                    return trainer;
                  });
  registerBuilder("PWCForestTrainer",
                  [](TiXmlNode * node)
                  {
                    Trainer * trainer = new PWCForestTrainer();
                    trainer->from_xml(node);
                    return trainer;
                  });
  registerBuilder("PWLForestTrainer",
                  [](TiXmlNode * node)
                  {
                    Trainer * trainer = new PWLForestTrainer();
                    trainer->from_xml(node);
                    return trainer;
                  });
}

}
