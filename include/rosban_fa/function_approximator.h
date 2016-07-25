#pragma once

#include "rosban_utils/serializable.h"

#include <Eigen/Core>

namespace rosban_fa
{

/// This class describe the interface of function approximators:
/// Function approximators can be trained to approximate a function
/// f: R^I -> R^O
/// where I is the dimension of the input space and O the dimension
/// of the output space
class FunctionApproximator
{
public:

  virtual ~FunctionApproximator();

  /// Return the output dimension of the function approximator
  virtual int getOutputDim() const = 0;

  /// Easy mapping for function approximator with O = 1
  /// throws a logic_error if O != 1 during the training phase
  void predict(const Eigen::VectorXd & input,
               double & mean,
               double & var) const;

  /// Predict the mean output and its covariance matrix for the given input
  virtual void predict(const Eigen::VectorXd & input,
                       Eigen::VectorXd & mean,
                       Eigen::MatrixXd & covar) const = 0;

  /// Compute the gradient of the approximation function at the given input
  /// throws a logic_error if output dim is not 1 during the training phase
  virtual void gradient(const Eigen::VectorXd & input,
                        Eigen::VectorXd & gradient) const = 0;

  /// Approximate argmax(f) and max(f) inside the provided limits
  /// throws a logic_error if output dim is not 1 during the training phase
  virtual void getMaximum(const Eigen::MatrixXd & limits,
                          Eigen::VectorXd & input,
                          double & output) const = 0;

  /// If not overriden, this method will launch an exception with the name of the class
  virtual void debugPrediction(const Eigen::VectorXd & input, std::ostream & out) const;

protected:

  /// Throws an explicit error message if the output is not 1D
  void check1DOutput(const std::string & caller_name) const;

};

}
