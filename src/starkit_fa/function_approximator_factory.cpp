#include "starkit_fa/function_approximator_factory.h"

#include "starkit_fa/constant_approximator.h"
#include "starkit_fa/fa_tree.h"
#include "starkit_fa/forest_approximator.h"
#include "starkit_fa/linear_approximator.h"
#include "starkit_fa/pwc_forest.h"
#include "starkit_fa/pwl_forest.h"

#ifdef STARKIT_FA_USES_DNN
#include "starkit_fa/dnn_approximator.h"
#endif
#ifdef STARKIT_FA_USES_GP
#include "starkit_fa/gp.h"
#include "starkit_fa/gp_forest.h"
#endif

namespace starkit_fa
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

#ifdef STARKIT_FA_USES_DNN
  registerBuilder(FunctionApproximator::DNNApproximator,
                  []() { return std::unique_ptr<FunctionApproximator>(new DNNApproximator); });
#endif
#ifdef STARKIT_FA_USES_GP
  registerBuilder(FunctionApproximator::GP, []() { return std::unique_ptr<FunctionApproximator>(new GP); });
  registerBuilder(FunctionApproximator::GPForest, []() { return std::unique_ptr<FunctionApproximator>(new GPForest); });
#endif
}

}  // namespace starkit_fa
