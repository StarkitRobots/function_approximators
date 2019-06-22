#include "starkit_fa/trainer.h"

#include "starkit_utils/serialization/factory.h"

namespace starkit_fa
{
class TrainerFactory : public starkit_utils::Factory<Trainer>
{
public:
  /// Automatically register several classes of Function Approximators Trainers
  TrainerFactory();
};

}  // namespace starkit_fa
