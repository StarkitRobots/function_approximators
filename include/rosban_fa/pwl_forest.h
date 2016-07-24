#pragma once

#include "rosban_fa/function_approximator.h"

#include "rosban_regression_forests/core/forest.h"

#include <Eigen/Core>

namespace rosban_fa
{

class PWLForest : public FunctionApproximator
{
public:

  virtual ~PWLForest();

  virtual int getOutputDim() const override;

  virtual void train(const Eigen::MatrixXd & input,
                     const Eigen::MatrixXd & observations,
                     const Eigen::MatrixXd & limits) override;

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
