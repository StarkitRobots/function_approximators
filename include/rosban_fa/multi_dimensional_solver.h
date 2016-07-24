#pragma once

#include "regression_experiments/solver.h"

#include <memory>

namespace regression_experiments
{

class MultiDimensionalSolver
{
  MultiDimensionalSolver(const std::string & solver_name);

  /// Update internal structure according to the provided samples
  /// for inputs: row = dimension, col = point_idx
  /// for observations: row = point_idx, col = dimension, 
  void solve(const Eigen::MatrixXd & inputs,
             const Eigen::MatrixXd & observations,
             const Eigen::MatrixXd & limits);

  /// Predict the outputs independently using internal structure
  /// for inputs: row = dimension, col = point_idx
  /// for means and vars: row = point_idx, col = dimension
  void predict(const Eigen::MatrixXd & inputs,
               Eigen::MatrixXd & means,
               Eigen::MatrixXd & vars);

private:
  std::string solver_name;
  std::vector<std::unique_ptr<Solver>> solvers;
};

}
