#pragma once

#include "rosban_fa/function_approximator.h"

#include "rosban_utils/serializable.h"

#include <Eigen/Core>

#include <functional>
#include <memory>
#include <random>

namespace rosban_fa
{

/// Optimizer trainers can be used to learn an approximation of the optimal
/// policy for a function requiring a set of parameters (which are imposed) and
/// a set of action (which can be chosen according to the parameters).
///
/// Formally:
/// - Let f be a (stochastic) blackbox function: P * A -> R
/// With:
/// P: the parameters of the function (imposed)
/// A: the action space (chosen)
/// R: the reward space (continuous)
///
/// The task of the optimizer is to implement a method which returns a function
/// approximator pi: S -> A, mapping states to estimation of optimal actions.
class OptimizerTrainer : public rosban_utils::Serializable
{
public:
  typedef std::function<double(const Eigen::VectorXd & parameters,
                               const Eigen::VectorXd & actions,
                               std::default_random_engine * engine)> RewardFunction;

  OptimizerTrainer();
  virtual ~OptimizerTrainer();

  /// Generate it's own sample to train on the given function
  virtual std::unique_ptr<FunctionApproximator>
  train(RewardFunction rf, std::default_random_engine * engine) = 0;

  /// Update the space of parameters
  void setParametersLimits(const Eigen::MatrixXd & new_limits);

  /// Update the space of actions
  void setActionsLimits(const Eigen::MatrixXd & new_limits);

protected:

  /// Number of dimensions for parameters
  int getParametersDim() const;

  /// Number of dimensions for actions
  int getActionsDim() const;

  /// Parameters space
  Eigen::MatrixXd parameters_limits;

  /// Action space
  Eigen::MatrixXd actions_limits;
};

}
