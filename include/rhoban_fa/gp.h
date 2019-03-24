#pragma once

#include "rhoban_fa/function_approximator.h"

#include "rhoban_gp/core/gaussian_process.h"
#include "rhoban_gp/gradient_ascent/randomized_rprop.h"

#include <memory>

namespace rhoban_fa
{
class GP : public FunctionApproximator
{
public:
  GP();
  GP(std::unique_ptr<std::vector<rhoban_gp::GaussianProcess>> gps, const rhoban_gp::RandomizedRProp::Config& ga_conf);

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
  std::unique_ptr<std::vector<rhoban_gp::GaussianProcess>> gps;
  rhoban_gp::RandomizedRProp::Config ga_conf;
};

}  // namespace rhoban_fa
