#pragma once

#include "starkit_fa/function_approximator.h"

#include "starkit_gp/core/gaussian_process.h"
#include "starkit_gp/gradient_ascent/randomized_rprop.h"

#include <memory>

namespace starkit_fa
{
class GP : public FunctionApproximator
{
public:
  GP();
  GP(std::unique_ptr<std::vector<starkit_gp::GaussianProcess>> gps, const starkit_gp::RandomizedRProp::Config& ga_conf);

  virtual ~GP();

  virtual std::unique_ptr<FunctionApproximator> clone() const override;

  virtual int getOutputDim() const override;

  virtual void predict(const Eigen::VectorXd& inputs, Eigen::VectorXd& mean, Eigen::MatrixXd& covar) const override;

  virtual void gradient(const Eigen::VectorXd& inputs, Eigen::VectorXd& gradients) const override;

  virtual void getMaximum(const Eigen::MatrixXd& limits, Eigen::VectorXd& input, double& output) const override;

  virtual int getClassID() const override;
  virtual int writeInternal(std::ostream& out) const override;
  virtual int read(std::istream& in) override;

private:
  /// One Gaussian process per output dimension
  std::unique_ptr<std::vector<starkit_gp::GaussianProcess>> gps;
  starkit_gp::RandomizedRProp::Config ga_conf;
};

}  // namespace starkit_fa
