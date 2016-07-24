#include "regression_experiments/benchmark_function_factory.h"
#include "regression_experiments/solver_factory.h"
#include "regression_experiments/tools.h"

#include "rosban_gp/scoring.h"

#include "rosban_utils/time_stamp.h"

#include "rosban_random/tools.h"

#include <fstream>
#include <memory>

using rosban_utils::TimeStamp;

namespace regression_experiments
{

Eigen::MatrixXd discretizeSpace(const Eigen::MatrixXd & limits,
                                const std::vector<int> & samples_by_dim)
{
  // Checking consistency
  if (limits.rows() != (int)samples_by_dim.size()) {
    throw std::runtime_error("discretizeSpace: inconsistency: limits.rows() != samples_by_dim");
  }
  // Preparing preliminary data
  int total_points = 1;
  std::vector<int> intervals(samples_by_dim.size());
  Eigen::VectorXd delta = limits.col(1) - limits.col(0);
  for (size_t dim = 0; dim < samples_by_dim.size(); dim++)
  {
    intervals[dim] = total_points;
    total_points *= samples_by_dim[dim];
  }
  // Preparing points
  Eigen::MatrixXd points(limits.rows(), total_points);
  for (int dim = 0; dim < limits.rows(); dim++) {
    for (int point = 0; point < total_points; point++) {
      // Determining index inside given dimension
      int dim_index = point / intervals[dim];
      dim_index = dim_index % samples_by_dim[dim];
      // Computing value
      double value = (limits(dim, 1) + limits(dim, 0)) / 2;// default value
      if (samples_by_dim[dim] != 1) {
        double step_size = delta(dim) / (samples_by_dim[dim] - 1);
        value = limits(dim, 0) + step_size * dim_index;
      }
      points(dim, point) = value;
    }
  }
  return points;
}

void buildPrediction(const std::string & function_name,
                     int nb_samples,
                     const std::string & solver_name,
                     const std::vector<int> & points_by_dim,
                     Eigen::MatrixXd & samples_inputs,
                     Eigen::VectorXd & samples_outputs,
                     Eigen::MatrixXd & prediction_points,
                     Eigen::VectorXd & prediction_means,
                     Eigen::VectorXd & prediction_vars,
                     Eigen::MatrixXd & gradients)
{
  // getting random engine
  auto engine = rosban_random::getRandomEngine();
  // Building function
  BenchmarkFunctionFactory bff;
  std::unique_ptr<BenchmarkFunction> benchmark_function(bff.build(function_name));
  // Generating random input
  benchmark_function->getUniformSamples(nb_samples, samples_inputs, samples_outputs, &engine);
  // Solving
  std::unique_ptr<Solver> solver(SolverFactory().build(solver_name));
  solver->solve(samples_inputs, samples_outputs, benchmark_function->getLimits());
  // Predicting
  prediction_points = discretizeSpace(benchmark_function->getLimits(), points_by_dim);
  solver->predict(prediction_points, prediction_means, prediction_vars);
  solver->gradients(prediction_points, gradients);
}

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
                  std::default_random_engine * engine)
{
  std::shared_ptr<Solver> solver(SolverFactory().build(solver_name));
  BenchmarkFunctionFactory bff;
  std::shared_ptr<BenchmarkFunction> function(bff.build(function_name));
  runBenchmark(function,
               nb_samples,
               solver,
               nb_test_points,
               smse,
               learning_time_ms,
               prediction_time_ms,
               arg_max_loss,
               max_prediction_error,
               compute_max_time_ms,
               engine);
}

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
                  std::default_random_engine * engine)
{
  // Internal data:
  Eigen::MatrixXd samples_inputs;
  Eigen::VectorXd samples_outputs;
  Eigen::MatrixXd test_points;
  Eigen::VectorXd test_observations, prediction_means, prediction_vars;
                    
  bool clean_engine = false;
  // getting random engine
  if (engine == NULL) {
    engine = rosban_random::newRandomEngine();
    clean_engine = true;
  }
  // Generating samples and test points
  function->getUniformSamples(nb_samples, samples_inputs, samples_outputs, engine);
  function->getUniformSamples(nb_test_points, test_points, test_observations, engine);
  // Solving
  TimeStamp learning_start = TimeStamp::now();
  solver->solve(samples_inputs, samples_outputs, function->getLimits());
  TimeStamp learning_end = TimeStamp::now();
  // Getting predictions for test points
  TimeStamp prediction_start = TimeStamp::now();
  solver->predict(test_points, prediction_means, prediction_vars);
  TimeStamp prediction_end = TimeStamp::now();
  // Evaluating prediction
  // Clean engine if necessary
  if (clean_engine) {
    delete(engine);
  }

  // Computing max
  Eigen::VectorXd best_input;
  double expected_max, measured_max;
  TimeStamp get_max_start = TimeStamp::now();
  solver->getMaximum(function->getLimits(), best_input, expected_max);
  TimeStamp get_max_end = TimeStamp::now();
  int nb_max_tests = 1000;
  measured_max = 0;
  for (int i = 0; i < nb_max_tests; i++) {
    measured_max += function->sample(best_input);
  }
  measured_max /= nb_max_tests;

  try{
    arg_max_loss = function->getMax() - measured_max;
    max_prediction_error = std::fabs(expected_max - measured_max);
  }
  catch(const std::runtime_error & exc) {
    arg_max_loss = -1;
    max_prediction_error = -1;
    std::cerr << exc.what() << std::endl;
  }

  // Computing output values
  smse = rosban_gp::computeSMSE(test_observations, prediction_means);
  learning_time_ms = diffMs(learning_start, learning_end);
  prediction_time_ms = diffMs(prediction_start, prediction_end);
  compute_max_time_ms = diffMs(get_max_start, get_max_end);

  double suspicion_min = std::pow(10,2);
  if (smse > suspicion_min) {
    std::cout << "Large smse: tracking debugs" << std::endl;
    for (int sample = 0; sample < nb_test_points; sample++) {
      double observation = test_observations(sample);
      double prediction = prediction_means(sample);
      double prediction_var = prediction_vars(sample);
      double diff2 = std::pow(observation - prediction, 2);
      if (diff2 > suspicion_min) {
        std::cout << "\tsample " << sample << ":" << std::endl;
        std::cout << "\t\tprediction : " << prediction  << std::endl
                  << "\t\tobservation: " << observation << std::endl
                  << "\t\tdiff2      : " << diff2       << std::endl;
        solver->debugPrediction(test_points.col(sample), std::cout);
      }
    }
  }
}

void writePrediction(const std::string & path,
                     const Eigen::MatrixXd & samples_inputs,
                     const Eigen::VectorXd & samples_outputs,
                     const Eigen::MatrixXd & prediction_points,
                     const Eigen::VectorXd & prediction_means,
                     const Eigen::VectorXd & prediction_vars,
                     const Eigen::MatrixXd & gradients)
{
  // Writing predictions + points
  std::ofstream out;
  out.open(path);
  out << "type,input,mean,min,max,gradient" << std::endl;

  // Writing Ref points
  for (int i = 0; i < samples_inputs.cols(); i++)
  {
    // write with the same format but min and max carry no meaning
    out << "observation," << samples_inputs(0,i) << ","
        << samples_outputs(i) << ",0,0,0" << std::endl;
  }

  // Writing predictions
  for (int point = 0; point < prediction_points.cols(); point++)
  {
    Eigen::VectorXd prediction_input = prediction_points.col(point);
    double mean = prediction_means(point);
    double var  = prediction_vars(point);
    // Getting +- 2 stddev
    double interval = 2 * std::sqrt(var);
    double min = mean - interval;
    double max = mean + interval;
    // Writing line
    out << "prediction," << prediction_input(0) << ","
        << mean << "," << min << "," << max << "," << gradients(0,point) << std::endl;
  }
  out.close();
}

}
