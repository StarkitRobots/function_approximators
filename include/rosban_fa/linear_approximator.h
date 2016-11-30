#pragma once

#include "rosban_fa/function_approximator.h"

namespace rosban_fa
{

/// This class implements approximation of a function with a linear model
/// prediction = A*X + B
class LinearApproximator : public FunctionApproximator
{
public:

  LinearApproximator();
  LinearApproximator(const Eigen::VectorXd & bias,
                     const Eigen::MatrixXd & coeffs);

  /// Parameters are ordered in the following way:
  /// bias, then coeffs.col(0), then coeffs.col(1), ...
  /// Throws a runtime_error if the number of parameters is not appropriate
  LinearApproximator(int input_dim, int output_dim,
                     const Eigen::VectorXd & parameters);

  virtual ~LinearApproximator();

  virtual int getOutputDim() const override;

  virtual void predict(const Eigen::VectorXd & input,
                       Eigen::VectorXd & mean,
                       Eigen::MatrixXd & covar) const override;

  virtual void gradient(const Eigen::VectorXd & input,
                        Eigen::VectorXd & gradient) const override;

  virtual void getMaximum(const Eigen::MatrixXd & limits,
                          Eigen::VectorXd & input,
                          double & output) const;

  virtual int getClassID() const override;
  virtual int writeInternal(std::ostream & out) const override;
  virtual int read(std::istream & in) override;

protected:

  /// bias: B in Y = A*X + B
  Eigen::VectorXd bias;

  /// coefficients of the squared matrix A in Y=A*X + B
  Eigen::MatrixXd coeffs;
};

}
