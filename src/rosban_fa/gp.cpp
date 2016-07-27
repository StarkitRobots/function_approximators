#include "rosban_fa/gp.h"

#include "rosban_gp/auto_tuning.h"
#include "rosban_gp/tools.h"
#include "rosban_gp/core/squared_exponential.h"

#include "rosban_random/tools.h"

#include <iostream>

using rosban_gp::GaussianProcess;

namespace rosban_fa
{

GP::GP()
{
}

GP::GP(std::unique_ptr<std::vector<rosban_gp::GaussianProcess>> gps_,
       const rosban_gp::RandomizedRProp::Config & ga_conf_)
  : gps(std::move(gps_)), ga_conf(ga_conf_)
{
}

GP::~GP() {}

int GP::getOutputDim() const
{
  if (!gps) return 0;
  return gps->size();
}


void GP::predict(const Eigen::VectorXd & input,
                       Eigen::VectorXd & mean,
                       Eigen::MatrixXd & covar) const
{
  int O = getOutputDim();
  mean = Eigen::VectorXd::Zero(O);
  covar = Eigen::MatrixXd::Identity(O, O);
  for (int output_dim = 0; output_dim < getOutputDim(); output_dim++) {
    double dim_mean, dim_var;
    (*gps)[output_dim].getDistribParameters(input, dim_mean, dim_var);
    mean(output_dim) = dim_mean;
    covar(output_dim, output_dim) = dim_var;
  }
}

void GP::gradient(const Eigen::VectorXd & input,
                  Eigen::VectorXd & gradient) const
{
  check1DOutput("gradient");
  gradient = Eigen::VectorXd::Zero(input.rows(), 1);
  gradient = (*gps)[0].getGradient(input);
}

void GP::getMaximum(const Eigen::MatrixXd & limits,
                          Eigen::VectorXd & input, double & output) const
{
  check1DOutput("getMaximum");
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
  best_guess = rosban_gp::RandomizedRProp::run(gradient_func, scoring_func, limits,
                                               ga_conf);
  input = best_guess;
  output = scoring_func(best_guess);
}

int GP::getClassID() const
{
  return FunctionApproximator::GP;
}

int GP::writeInternal(std::ostream & out) const
{
  int bytes_written = 0;
  bytes_written += rosban_utils::write<int>(out, getOutputDim());
  for (int dim = 0; dim < getOutputDim(); dim++) {
    bytes_written += (*gps)[dim].write(out);//TODO: change to write internal if it is changed
  }
  return bytes_written;
}

int GP::read(std::istream & in)
{
  int bytes_read = 0;
  int output_dims;
  bytes_read += rosban_utils::read<int>(in, &output_dims);
  gps = std::unique_ptr<std::vector<GaussianProcess>>(new std::vector<GaussianProcess>(output_dims));
  for (int output_dim = 0; output_dim < output_dims; output_dim++) {
    bytes_read += (*gps)[output_dim].read(in);
  }
  return bytes_read;
}

}
