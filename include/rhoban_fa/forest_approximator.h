#pragma once

#include "rhoban_fa/function_approximator.h"

#include "rhoban_regression_forests/core/forest.h"

namespace rhoban_fa
{

/// A generic class for forest approximatiors
class ForestApproximator : public FunctionApproximator
{
public:
  typedef std::vector<std::unique_ptr<regression_forests::Forest>> Forests;

  ForestApproximator();
  ForestApproximator(std::unique_ptr<Forests> forests,
                     int max_action_tiles);

  virtual ~ForestApproximator();

  virtual std::unique_ptr<FunctionApproximator> clone() const override;

  void setForests(std::unique_ptr<Forests> forests);
  void setMaxActionTiles(int max_action_tiles);
  void setAggregationMethod(regression_forests::Forest::AggregationMethod aggregation_method);

  virtual int getOutputDim() const override;

  /// Predict the outputs independently using internal structure
  virtual void predict(const Eigen::VectorXd & input,
                       Eigen::VectorXd & mean,
                       Eigen::MatrixXd & covar) const override;

  virtual void gradient(const Eigen::VectorXd & input,
                        Eigen::VectorXd & gradient) const override;

  virtual void getMaximum(const Eigen::MatrixXd & limits,
                          Eigen::VectorXd & input,
                          double & output) const override;

  virtual int getClassID() const override;
  virtual int writeInternal(std::ostream & out) const override;
  virtual int read(std::istream & in) override;

  static std::unique_ptr<Forests> cloneForests(const Forests & f);

protected:
  std::unique_ptr<Forests> forests;
  int max_action_tiles;
  regression_forests::Forest::AggregationMethod aggregation_method;
};

}
