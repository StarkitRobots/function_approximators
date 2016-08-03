#include "rosban_fa/function_approximator_factory.h"

#include "rosban_fa/forest_approximator.h"
#include "rosban_fa/gp.h"
#include "rosban_fa/gp_forest.h"
#include "rosban_fa/pwc_forest.h"
#include "rosban_fa/pwl_forest.h"

namespace rosban_fa
{

FunctionApproximatorFactory::FunctionApproximatorFactory()
{
  registerBuilder(FunctionApproximator::GP,
                  []() { return std::unique_ptr<FunctionApproximator>(new GP); });
  registerBuilder(FunctionApproximator::ForestApproximator,
                  []() { return std::unique_ptr<FunctionApproximator>(new ForestApproximator); });
  registerBuilder(FunctionApproximator::GPForest,
                  []() { return std::unique_ptr<FunctionApproximator>(new GPForest); });
  registerBuilder(FunctionApproximator::PWCForest,
                  []() { return std::unique_ptr<FunctionApproximator>(new PWCForest); });
  registerBuilder(FunctionApproximator::PWLForest,
                  []() { return std::unique_ptr<FunctionApproximator>(new PWLForest); });
}

}
