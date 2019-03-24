#pragma once

#include "rhoban_fa/forest_approximator.h"

#include "rhoban_gp/gradient_ascent/randomized_rprop.h"

namespace rhoban_fa
{
class GPForest : public ForestApproximator
{
public:
  typedef std::vector<std::unique_ptr<regression_forests::Forest>> Forests;

  GPForest();
  GPForest(std::unique_ptr<Forests> forests, const rhoban_gp::RandomizedRProp::Config& conf);

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
  rhoban_gp::RandomizedRProp::Config ga_conf;
};

}  // namespace rhoban_fa
