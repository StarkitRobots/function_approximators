#pragma once

#include "rosban_fa/function_approximator.h"
#include "rosban_fa/split.h"

#include <memory>

namespace rosban_fa
{

/// This class is a FunctionApproximator hub, in which every child is a
/// FunctionApproximator itself. It is mainly used to represent trees of
/// FunctionApproximator. Each child of the hub is used to approximate a
/// subset of the input space
class FATree : public FunctionApproximator
{
public:
  FATree();
  virtual ~FATree();

  int getOutputDim() const override;

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

protected:

  /// Throws an explicit exception if content was not properly set
  void checkConsistency(const std::string & caller_name) const;

  /// The input space is separated in several parts (eventually more than two)
  std::unique_ptr<Split> split;
  /// List of the childs which can be used for the tree
  std::vector<std::unique_ptr<FunctionApproximator>> childs;

};


}
