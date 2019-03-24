#include "rhoban_fa/optimizer_trainer.h"

namespace rhoban_fa
{
OptimizerTrainer::OptimizerTrainer()
{
}

OptimizerTrainer::~OptimizerTrainer()
{
}

void OptimizerTrainer::reset()
{
}

void OptimizerTrainer::setParametersLimits(const Eigen::MatrixXd& new_limits)
{
  parameters_limits = new_limits;
}

void OptimizerTrainer::setActionsLimits(const Eigen::MatrixXd& new_limits)
{
  actions_limits = new_limits;
}

void OptimizerTrainer::setNbThreads(int nb_threads_)
{
  nb_threads = nb_threads_;
}

int OptimizerTrainer::getParametersDim() const
{
  return parameters_limits.rows();
}

int OptimizerTrainer::getActionsDim() const
{
  return actions_limits.rows();
}

}  // namespace rhoban_fa
