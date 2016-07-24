#pragma once

#include "rosban_fa/function_approximator.h"

#include "rosban_gp/core/gaussian_process.h"
#include "rosban_gp/gradient_ascent/randomized_rprop.h"

#include <memory>

namespace rosban_fa
{

class GP : public FunctionApproximator
{
public:

  GP(std::unique_ptr<std::vector<rosban_gp::GaussianProcess>> gps,
     const rosban_gp::RandomizedRProp::Config & ga_conf);

  virtual ~GP();

  virtual int getOutputDim() const override;

  virtual void predict(const Eigen::VectorXd & inputs,
                       Eigen::VectorXd & mean,
                       Eigen::MatrixXd & covar) const override;

  virtual void gradient(const Eigen::VectorXd & inputs,
                        Eigen::VectorXd & gradients) const override;

  virtual void getMaximum(const Eigen::MatrixXd & limits,
                          Eigen::VectorXd & input,
                          double & output) const override;

private:
  /// One Gaussian process per output dimension
  std::unique_ptr<std::vector<rosban_gp::GaussianProcess>> gps;
  rosban_gp::RandomizedRProp::Config ga_conf;
};

}
