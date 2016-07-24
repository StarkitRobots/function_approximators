#pragma once

#include "rosban_fa/function_approximator.h"

#include "rosban_regression_forests/core/forest.h"

#include "rosban_gp/gradient_ascent/randomized_rprop.h"

#include <Eigen/Core>

namespace rosban_fa
{

class GPForest : public FunctionApproximator
{
public:
  /// Which type of GPForest is used
  /// - SQRT: |samples|^{1/2} samples per node min
  /// - CURT: |samples|^{1/3} samples per node min
  /// - LOG2: log_2(|samples|)  sampels per node min
  enum class Type
  { SQRT, CURT, LOG2};

  GPForest(Type t = Type::LOG2);

  virtual ~GPForest();

  virtual int getOutputDim() const override;

  virtual void train(const Eigen::MatrixXd & inputs,
                     const Eigen::MatrixXd & observations,
                     const Eigen::MatrixXd & limits) override;

  virtual void predict(const Eigen::VectorXd & input,
                       Eigen::VectorXd & means,
                       Eigen::MatrixXd & covar) override;

  virtual void debugPrediction(const Eigen::VectorXd & input, std::ostream & out) override;

  virtual void gradient(const Eigen::VectorXd & inputs,
                        Eigen::VectorXd & gradient) override;

  virtual void getMaximum(const Eigen::MatrixXd & limits,
                          Eigen::VectorXd & input, double & output) override;

  virtual std::string class_name() const override;
  virtual void to_xml(std::ostream &out) const override;
  virtual void from_xml(TiXmlNode *node) override;

private:
  std::vector<std::unique_ptr<regression_forests::Forest>> forests;
  Type type;
  int nb_threads;

  rosban_gp::RandomizedRProp::Config approximation_conf;
  rosban_gp::RandomizedRProp::Config find_max_conf;
};
GPForest::Type loadType(const std::string &s);
std::string to_string(GPForest::Type type);

}
