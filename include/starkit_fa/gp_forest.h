#pragma once

#include "starkit_fa/forest_approximator.h"

#include "starkit_gp/gradient_ascent/randomized_rprop.h"

namespace starkit_fa
{
class GPForest : public ForestApproximator
{
public:
  typedef std::vector<std::unique_ptr<regression_forests::Forest>> Forests;

  GPForest();
  GPForest(std::unique_ptr<Forests> forests, const starkit_gp::RandomizedRProp::Config& conf);

  virtual ~GPForest();

  virtual std::unique_ptr<FunctionApproximator> clone() const override;

  virtual void predict(const Eigen::VectorXd& input, Eigen::VectorXd& means, Eigen::MatrixXd& covar) const override;

  virtual void debugPrediction(const Eigen::VectorXd& input, std::ostream& out) const override;

  virtual void gradient(const Eigen::VectorXd& inputs, Eigen::VectorXd& gradient) const override;

  virtual void getMaximum(const Eigen::MatrixXd& limits, Eigen::VectorXd& input, double& output) const override;

  virtual int getClassID() const override;
  virtual int writeInternal(std::ostream& out) const override;
  virtual int read(std::istream& in) override;

private:
  starkit_gp::RandomizedRProp::Config ga_conf;
};

}  // namespace starkit_fa
