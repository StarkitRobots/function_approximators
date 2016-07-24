#pragma once

#include "regression_experiments/benchmark_function.h"
#include "regression_experiments/solver.h"

#include <Eigen/Core>

#include <memory>

namespace regression_experiments
{

/// Return a matrix containing product(samples_by_dim) columns and limits.rows() rows
/// Each column is a different sample
Eigen::MatrixXd discretizeSpace(const Eigen::MatrixXd & limits,
                                const std::vector<int> & samples_by_dim);

/// 1. Create random samples of the given function
/// 2. Solve them using the chosen solver
/// 3. Predict the output on the given grid
void buildPrediction(const std::string & function_name,
                     int nb_samples,
                     const std::string & solver_name,
                     const std::vector<int> & points_by_dim,
                     Eigen::MatrixXd & samples_inputs,
                     Eigen::VectorXd & samples_outputs,
                     Eigen::MatrixXd & prediction_points,
                     Eigen::VectorXd & prediction_means,
                     Eigen::VectorXd & prediction_vars,
                     Eigen::MatrixXd & gradients);

/// 1. Generate learning and test samples for the given function
/// 2. Create a regression model using the chosen solver and the generated samples
/// 3. Evaluate the quality of the regression model using the test set
void runBenchmark(const std::string & function_name,
                  int nb_samples,
                  const std::string & solver_name,
                  int nb_test_points,
                  double & smse,
                  double & learning_time_ms,
                  double & prediction_time_ms,
                  double & arg_max_loss,
                  double & max_prediction_error,
                  double & compute_max_time_ms,
                  std::default_random_engine * engine);

void runBenchmark(std::shared_ptr<BenchmarkFunction> function,
                  int nb_samples,
                  std::shared_ptr<Solver> solver,
                  int nb_test_points,
                  double & smse,
                  double & learning_time_ms,
                  double & prediction_time_ms,
                  double & arg_max_loss,
                  double & max_prediction_error,
                  double & compute_max_time_ms,
                  std::default_random_engine * engine);

void writePrediction(const std::string & path,
                     const Eigen::MatrixXd & samples_inputs,
                     const Eigen::VectorXd & samples_outputs,
                     const Eigen::MatrixXd & prediction_points,
                     const Eigen::VectorXd & prediction_means,
                     const Eigen::VectorXd & prediction_vars,
                     const Eigen::MatrixXd & gradients);

}
