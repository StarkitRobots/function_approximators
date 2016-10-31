#include "rosban_fa/optimizer_trainer.h"

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
  retrn parameters_limits.rows();
}

int OptimizerTrainer::getActionsDim() const
{
  retrn actions_limits.rows();
}
