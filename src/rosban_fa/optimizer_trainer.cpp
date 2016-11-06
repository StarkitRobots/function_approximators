#include "rosban_fa/optimizer_trainer.h"

namespace rosban_fa
{

OptimizerTrainer::OptimizerTrainer() {}

OptimizerTrainer::~OptimizerTrainer() {}

void OptimizerTrainer::setParametersLimits(const Eigen::MatrixXd & new_limits)
{
  parameters_limits = new_limits;
}

void OptimizerTrainer::setActionsLimits(const Eigen::MatrixXd & new_limits)
{
  actions_limits = new_limits;
}

int OptimizerTrainer::getParametersDim() const
{
  return parameters_limits.rows();
}

int OptimizerTrainer::getActionsDim() const
{
  return actions_limits.rows();
}

}
