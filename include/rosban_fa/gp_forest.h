#pragma once

#include "rosban_fa/function_approximator.h"

#include "rosban_regression_forests/core/forest.h"

#include "rosban_gp/gradient_ascent/randomized_rprop.h"

namespace rosban_fa
{

class GPForest : public FunctionApproximator
{
public:

  typedef std::vector<std::unique_ptr<regression_forests::Forest>> Forests;

  GPForest();

  virtual ~GPForest();

  virtual int getOutputDim() const override;

  virtual void predict(const Eigen::VectorXd & input,
                       Eigen::VectorXd & means,
                       Eigen::MatrixXd & covar) const override;

  virtual void debugPrediction(const Eigen::VectorXd & input,
                               std::ostream & out) const override;

  virtual void gradient(const Eigen::VectorXd & inputs,
                        Eigen::VectorXd & gradient) const override;

  virtual void getMaximum(const Eigen::MatrixXd & limits,
                          Eigen::VectorXd & input,
                          double & output) const override;

private:
  std::unique_ptr<Forests> forests;
  rosban_gp::RandomizedRProp::Config ga_conf;
};

}
