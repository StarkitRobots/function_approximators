#include "rosban_fa/trainer.h"

#include "rhoban_utils/serialization/factory.h"

namespace rosban_fa
{

class TrainerFactory : public rhoban_utils::Factory<Trainer>
{
public:
  /// Automatically register several classes of Function Approximators Trainers
  TrainerFactory();
};


}
