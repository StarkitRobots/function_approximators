#include "rosban_fa/function_approximator.h"

#include "rosban_utils/factory.h"

namespace rosban_fa
{

class Factory : public rosban_utils::Factory<FunctionApproximator>
{
public:
  /// Automatically register several classes of Function Approximators
  Factory();
};


}
