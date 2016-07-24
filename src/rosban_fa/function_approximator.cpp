#include "rosban_fa/function_approximator.h"

#include <sstream>
#include <stdexcept>

namespace rosban_fa
{


void FunctionApproximator::predict(const Eigen::VectorXd & input,
                                   double & mean,
                                   double & var)
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

void FunctionApproximator::debugPrediction(const Eigen::VectorXd & input, std::ostream & out)
{
  (void)input;(void)out;
  std::ostringstream oss;
  oss << "FunctionApproximator::debugPrediction: "
      << "Unimplemented Method for class '" << class_name() << "'";
  throw std::logic_error(oss.str());
}

void FunctionApproximator::checkConsistency(const Eigen::MatrixXd & inputs,
                                            const Eigen::MatrixXd & observations,
                                            const Eigen::MatrixXd & limits)
{
  if (inputs.cols() != observations.rows()) {
    std::ostringstream oss;
    oss << "FunctionApproximator::checkConsistency: inconsistent number of samples: "
        << "inputs.cols() != observations.rows() "
        << "(" << inputs.cols() << " != " << observations.rows() << ")";
    throw std::logic_error(oss.str());
  }
  if (limits.rows() != inputs.rows()) {
    std::ostringstream oss;
    oss << "FunctionApproximator::checkConsistency: inconsistent dimension for input: "
        << "inputs.rows() != limits.rows() "
        << "(" << inputs.rows() << " != " << limits.rows() << ")";
    throw std::logic_error(oss.str());
  }
  if (limits.cols() != 2) {
    std::ostringstream oss;
    oss << "FunctionApproximator::checkConsistency: invalid dimensions for limits: "
        << "expecting 2 columns and received " << limits.rows();
    throw std::logic_error(oss.str());
  }
}

void FunctionApproximator::check1DOutput(const std::string & caller_name)
{
  if (getOutputDim() != 1) {
    std::ostringstream oss;
    oss << class_name() << "::" << caller_name << ": "
        << "requiring a 1D output to a "
        << getOutputDim() << "D output function";
    throw std::logic_error(oss.str());
  }
}

}
