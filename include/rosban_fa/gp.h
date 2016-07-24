#pragma once

#include "rosban_fa/function_approximator.h"

#include "rosban_gp/core/gaussian_process.h"
#include "rosban_gp/gradient_ascent/randomized_rprop.h"


#include <Eigen/Core>

namespace rosban_fa
{

class GP : public FunctionApproximator
{
public:

  virtual ~GP() {}

  /// Update internal structure according to the provided samples
  virtual void train(const Eigen::MatrixXd & inputs,
                     const Eigen::MatrixXd & observations,
                     const Eigen::MatrixXd & limits) override;

  /// Predict the outputs independently using internal structure
  virtual void predict(const Eigen::VectorXd & inputs,
                       Eigen::VectorXd & mean,
                       Eigen::MatrixXd & covar) override;

  virtual void gradient(const Eigen::VectorXd & inputs,
                        Eigen::VectorXd & gradients) override;

  virtual void getMaximum(const Eigen::MatrixXd & limits,
                          Eigen::VectorXd & input,
                          double & output) override;

  virtual std::string class_name() const override;
  virtual void to_xml(std::ostream &out) const override;
  virtual void from_xml(TiXmlNode *node) override;

private:
  /// One Gaussian process per output dimension
  std::vector<rosban_gp::GaussianProcess> gps;
  rosban_gp::RandomizedRProp::Config conf;
};

}
