#include "regression_experiments/solver.h"

#include "rosban_utils/factory.h"

namespace regression_experiments
{

class SolverFactory : public rosban_utils::Factory<Solver>
{
public:
  SolverFactory();
};


}
