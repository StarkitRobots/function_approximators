#include "rosban_fa/function_approximator.h"

#include <sstream>
#include <stdexcept>

namespace rosban_fa
{

FunctionApproximator::~FunctionApproximator() {}

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

}
