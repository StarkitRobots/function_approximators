#pragma once

#include "rosban_utils/serializable.h"

#include <Eigen/Core>

#include <functional>
#include <memory>

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
  typedef std::function<double(Eigen::VectorXd parameters,
                               Eigen::VectorXd actions,
                               std::default_random_engine * engine)> RewardFunction;

  OptimizerTrainer();
  virtual ~OptimizerTrainer();

  virtual std::unique_ptr<FunctionApproximator>
  train(RewardFunction rf);
};
