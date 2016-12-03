#pragma once

#include "rosban_fa/function_approximator.h"

namespace rosban_fa
{

/// This class implements approximation of a function by a constant
class ConstantApproximator : public FunctionApproximator
{
public:

  ConstantApproximator();
  ConstantApproximator(const Eigen::VectorXd & average);

  virtual ~ConstantApproximator();

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

  virtual std::string toString() const override;

protected:

  /// Approximation is a constant
  Eigen::VectorXd average;
};

}
