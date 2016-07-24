#include "rosban_fa/gp.h"

#include "rosban_gp/auto_tuning.h"
#include "rosban_gp/tools.h"
#include "rosban_gp/core/squared_exponential.h"

#include "rosban_random/tools.h"

#include <iostream>

using rosban_gp::CovarianceFunction;
using rosban_gp::GaussianProcess;
using rosban_gp::SquaredExponential;

namespace rosban_fa
{

void GP::train(const Eigen::MatrixXd & inputs,
               const Eigen::MatrixXd & observations,
               const Eigen::MatrixXd & limits)
{
  checkConsistency(inputs, observations, limits);
  gps.clear();
  for (int output_dim = 0; output_dim < observations.cols(); output_dim++)
  {
    std::unique_ptr<CovarianceFunction> cov_func(new SquaredExponential(inputs.rows()));
    gps.push_back(GaussianProcess(inputs, observations.col(output_dim),
                                  std::move(cov_func)));
    gps[output_dim].autoTune(conf);
  }
}

void GP::predict(const Eigen::VectorXd & input,
                       Eigen::VectorXd & mean,
                       Eigen::MatrixXd & covar)
{
  mean = Eigen::VectorXd::Zero(gps.size());
  Eigen::VectorXd vars = Eigen::VectorXd::Zero(gps.size());
  for (size_t output_dim = 0; output_dim < gps.size(); output_dim++) {
    double dim_mean, dim_var;
    gps[output_dim].getDistribParameters(input, dim_mean, dim_var);
    mean(output_dim) = dim_mean;
    vars(output_dim) = dim_var;
  }
  covar = Eigen::MatrixXd::Identity(gps.size(), gps.size()) * vars;
}

void GP::gradient(const Eigen::VectorXd & input,
                  Eigen::VectorXd & gradient)
{
  check1DOutput("gradient");
  gradient = Eigen::VectorXd::Zero(input.rows(), 1);
  gradient = gps[0].getGradient(input);
}

void GP::getMaximum(const Eigen::MatrixXd & limits,
                          Eigen::VectorXd & input, double & output)
{
  check1DOutput("getMaximum");
  // TODO put those parameters as parseable in xml
  // rProp properties
  int nb_trials = 100;
  double epsilon = std::pow(10, -6);
  int max_nb_guess = 2000;
  // Preparing functions
  std::function<Eigen::VectorXd(const Eigen::VectorXd)> gradient_func;
  gradient_func = [this](const Eigen::VectorXd & guess)
    {
      Eigen::VectorXd gradient;
      this->gradient(guess, gradient);
      return gradient;
    };
  std::function<double(const Eigen::VectorXd)> scoring_func;
  scoring_func = [this](const Eigen::VectorXd & guess)
    {
      Eigen::VectorXd values;
      Eigen::MatrixXd covar;
      this->predict(guess, values, covar);
      return values(0);
    };
  // Performing multiple rProp and conserving the best candidate
  Eigen::VectorXd best_guess;
  best_guess = rosban_gp::randomizedRProp(gradient_func, scoring_func, limits,
                                          epsilon, nb_trials, max_nb_guess);
  input = best_guess;
  output = scoring_func(best_guess);
}

std::string GP::class_name() const
{
  return "gp";
}

void GP::to_xml(std::ostream &out) const
{
  conf.write("conf", out);
}

void GP::from_xml(TiXmlNode *node)
{
  conf.tryRead(node, "conf");
}

}
