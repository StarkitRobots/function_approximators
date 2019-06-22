#pragma once

#include "starkit_utils/serialization/stream_serializable.h"

#include <Eigen/Core>

#include <memory>

namespace starkit_fa
{
/// This class describe the interface of function approximators:
/// Function approximators can be trained to approximate a function
/// f: R^I -> R^O
/// where I is the dimension of the input space and O the dimension
/// of the output space
class FunctionApproximator : public starkit_utils::StreamSerializable
{
public:
  virtual ~FunctionApproximator();

  virtual std::unique_ptr<FunctionApproximator> clone() const = 0;

  /// Return the output dimension of the function approximator
  virtual int getOutputDim() const = 0;

  // Return the predicted output for the given input
  Eigen::VectorXd predict(const Eigen::VectorXd& input) const;

  /// Default implementation computes the prediction with variance and
  /// discard the variance information
  /// - might be overrided for speedup
  virtual double predict(const Eigen::VectorXd& input, int dim) const;

  /// Easy mapping for function approximator with O = 1
  /// throws a logic_error if O != 1 during the training phase
  void predict(const Eigen::VectorXd& input, double& mean, double& var) const;

  /// Default implementation computes the global prediction and then
  /// select only the relevant features
  /// - might be overrided for speedup
  virtual void predict(const Eigen::VectorXd& input, int dim, double& mean, double& var) const;

  /// Predict the mean output and its covariance matrix for the given input
  virtual void predict(const Eigen::VectorXd& input, Eigen::VectorXd& mean, Eigen::MatrixXd& covar) const = 0;

  /// Compute the gradient of the approximation function at the given input
  /// throws a logic_error if output dim is not 1 during the training phase
  virtual void gradient(const Eigen::VectorXd& input, Eigen::VectorXd& gradient) const = 0;

  /// Approximate argmax(f) and max(f) inside the provided limits
  /// throws a logic_error if output dim is not 1 during the training phase
  virtual void getMaximum(const Eigen::MatrixXd& limits, Eigen::VectorXd& input, double& output) const = 0;

  /// If not overriden, this method will launch an exception with the name of the class
  virtual void debugPrediction(const Eigen::VectorXd& input, std::ostream& out) const;

  /// Convert the function approximator to a string, default behavior only shows
  /// the identifier of the class
  virtual std::string toString() const;

protected:
  /// Throws an explicit error message if the output is not 1D
  void check1DOutput(const std::string& caller_name) const;

public:
  /// Each function_approximator has its own class_id
  enum ID : int
  {
    GP = 1,
    GPForest = 2,
    PWCForest = 3,
    PWLForest = 4,
    ForestApproximator = 5,
    FATree = 6,
    Constant = 7,
    Linear = 8,
    DNNApproximator = 9
  };
};

}  // namespace starkit_fa
