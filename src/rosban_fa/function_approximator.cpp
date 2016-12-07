#include "rosban_fa/function_approximator.h"

#include <sstream>
#include <stdexcept>

namespace rosban_fa
{

FunctionApproximator::~FunctionApproximator() {}

Eigen::VectorXd FunctionApproximator::predict(const Eigen::VectorXd & input) const
{
  Eigen::VectorXd mean;
  Eigen::MatrixXd covar;
  predict(input, mean, covar);
  return mean;
}

double FunctionApproximator::predict(const Eigen::VectorXd & input, int dim) const
{
  double mean, var;
  predict(input, dim, mean, var);
  return mean;
}

void FunctionApproximator::predict(const Eigen::VectorXd & input,
                                   int dim,
                                   double & mean,
                                   double & var) const
{
  Eigen::VectorXd means;
  Eigen::MatrixXd covar;
  predict(input, means, covar);
  mean = means(dim);
  var = covar(dim, dim);
}

void FunctionApproximator::predict(const Eigen::VectorXd & input,
                                   double & mean,
                                   double & var) const
{
  Eigen::VectorXd output;
  Eigen::MatrixXd covar;
  predict(input, output, covar);
  if (output.rows() != 1) {
    std::ostringstream oss;
    oss << "FunctionApproximator::predict: requiring a 1D output to a "
        << output.rows() << "D output function";
    throw std::logic_error(oss.str());
  }

  mean = output(0);
  var = covar(0,0);
}

void FunctionApproximator::debugPrediction(const Eigen::VectorXd & input, std::ostream & out) const
{
  (void)input;(void)out;
  std::ostringstream oss;
  oss << "FunctionApproximator::debugPrediction: "
      << "Unimplemented Method for current class";
  throw std::logic_error(oss.str());
}

void FunctionApproximator::check1DOutput(const std::string & caller_name) const
{
  if (getOutputDim() != 1) {
    std::ostringstream oss;
    oss << "FunctionApproximator" << "::" << caller_name << ": "
        << "requiring a 1D output to a "
        << getOutputDim() << "D output function";
    throw std::logic_error(oss.str());
  }
}

std::string FunctionApproximator::toString() const {
  std::ostringstream oss;
  oss << "(FunctionApproximator: classID=" << getClassID() << "| internal: unimplemented)";
  return oss.str();
}

}
