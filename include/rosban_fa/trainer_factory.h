#include "rosban_fa/trainer.h"

#include "rosban_utils/factory.h"

namespace rosban_fa
{

class TrainerFactory : public rosban_utils::Factory<Trainer>
{
public:
  /// Automatically register several classes of Function Approximators Trainers
  TrainerFactory();
};


}
