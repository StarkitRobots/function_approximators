#pragma once

#include "rosban_utils/serializable.h"

#include <Eigen/Core>

#include <ostream>
#include <stdexcept>

namespace rosban_fa
{

/// This class describe the interface of function approximators:
/// Function approximators can be trained to approximate a function
/// f: R^I -> R^O
/// where I is the dimension of the input space and O the dimension
/// of the output space
///
/// All function approximators can be serialized as xml files in order to
/// allow loading their configuration from a file
class FunctionApproximator : public rosban_utils::Serializable
{
public:

  virtual ~FunctionApproximator() {}

  virtual int getOutputDim() const = 0;

  /// Train the function approximator with the provided set of N samples
  /// inputs: a I by N matrix, each column is a different input
  /// outputs: a N by O matrix, each row is a different output 
  /// limits:
  /// - column 0 is min
  /// - column 1 is max
  virtual void train(const Eigen::MatrixXd & inputs,
                     const Eigen::MatrixXd & observations,
                     const Eigen::MatrixXd & limits) = 0;

  /// Easy mapping for function approximator with O = 1
  /// throws a logic_error if O != 1 during the training phase
  void predict(const Eigen::VectorXd & input,
               double & mean,
               double & var);

  /// Predict the mean output and its covariance matrix for the given input
  virtual void predict(const Eigen::VectorXd & input,
                       Eigen::VectorXd & mean,
                       Eigen::MatrixXd & covar) = 0;

  /// Compute the gradient of the approximation function at the given input
  /// throws a logic_error if output dim is not 1 during the training phase
  virtual void gradient(const Eigen::VectorXd & input,
                        Eigen::VectorXd & gradient) = 0;

  /// Approximate argmax(f) and max(f) inside the provided limits
  /// throws a logic_error if output dim is not 1 during the training phase
  virtual void getMaximum(const Eigen::MatrixXd & limits,
                          Eigen::VectorXd & input,
                          double & output) = 0;

  /// If not overriden, this method will launch an exception with the name of the class
  virtual void debugPrediction(const Eigen::VectorXd & input, std::ostream & out);

protected:

  /// Throws an explicit logic_error if informations are not consistent
  void checkConsistency(const Eigen::MatrixXd & inputs,
                        const Eigen::MatrixXd & observations,
                        const Eigen::MatrixXd & limits);

  /// Throws an explicit error message if the output is not 1D
  void check1DOutput(const std::string & caller_name);

};

}
