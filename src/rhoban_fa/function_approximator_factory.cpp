#include "rhoban_fa/function_approximator_factory.h"

#include "rhoban_fa/constant_approximator.h"
#include "rhoban_fa/fa_tree.h"
#include "rhoban_fa/forest_approximator.h"
#include "rhoban_fa/linear_approximator.h"
#include "rhoban_fa/pwc_forest.h"
#include "rhoban_fa/pwl_forest.h"

#ifdef RHOBAN_FA_USES_DNN
#include "rhoban_fa/dnn_approximator.h"
#endif
#ifdef RHOBAN_FA_USES_GP
#include "rhoban_fa/gp.h"
#include "rhoban_fa/gp_forest.h"
#endif

namespace rhoban_fa
{
FunctionApproximatorFactory::FunctionApproximatorFactory()
{
  registerBuilder(FunctionApproximator::ForestApproximator,
                  []() { return std::unique_ptr<FunctionApproximator>(new ForestApproximator); });
  registerBuilder(FunctionApproximator::PWCForest,
                  []() { return std::unique_ptr<FunctionApproximator>(new PWCForest); });
  registerBuilder(FunctionApproximator::PWLForest,
                  []() { return std::unique_ptr<FunctionApproximator>(new PWLForest); });
  registerBuilder(FunctionApproximator::FATree, []() { return std::unique_ptr<FunctionApproximator>(new FATree); });
  registerBuilder(FunctionApproximator::Constant,
                  []() { return std::unique_ptr<FunctionApproximator>(new ConstantApproximator); });
  registerBuilder(FunctionApproximator::Linear,
                  []() { return std::unique_ptr<FunctionApproximator>(new LinearApproximator); });

#ifdef RHOBAN_FA_USES_DNN
  registerBuilder(FunctionApproximator::DNNApproximator,
                  []() { return std::unique_ptr<FunctionApproximator>(new DNNApproximator); });
#endif
#ifdef RHOBAN_FA_USES_GP
  registerBuilder(FunctionApproximator::GP, []() { return std::unique_ptr<FunctionApproximator>(new GP); });
  registerBuilder(FunctionApproximator::GPForest, []() { return std::unique_ptr<FunctionApproximator>(new GPForest); });
#endif
}

}  // namespace rhoban_fa
