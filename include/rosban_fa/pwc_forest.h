#pragma once

#include "rosban_fa/function_approximator.h"

#include "rosban_regression_forests/core/forest.h"

#include <Eigen/Core>

namespace rosban_fa
{

class PWCForest : public FunctionApproximator
{
public:

  virtual ~PWCForest();

  virtual int getOutputDim() const override;

  /// Update internal structure according to the provided samples
  virtual void train(const Eigen::MatrixXd & inputs,
                     const Eigen::MatrixXd & observations,
                     const Eigen::MatrixXd & limits) override;

  /// Predict the outputs independently using internal structure
  virtual void predict(const Eigen::VectorXd & input,
                       Eigen::VectorXd & mean,
                       Eigen::MatrixXd & covar) override;

  virtual void gradient(const Eigen::VectorXd & input,
                        Eigen::VectorXd & gradient) override;

  virtual void getMaximum(const Eigen::MatrixXd & limits,
                          Eigen::VectorXd & input,
                          double & output) override;

  virtual std::string class_name() const override;
  virtual void to_xml(std::ostream &out) const override;
  virtual void from_xml(TiXmlNode *node) override;

private:
  std::vector<std::unique_ptr<regression_forests::Forest>> forests;
};

}
