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
  /// Transmit ownership and clear vector 'childs'
  FATree(std::unique_ptr<Split> split,
         std::vector<std::unique_ptr<FunctionApproximator>> & childs);
  virtual ~FATree();

  std::unique_ptr<FATree> clone() const;

  int getOutputDim() const override;

  /// Retrieve the elemental function approximator used at this point
  const FunctionApproximator &
  getLeafApproximator(const Eigen::VectorXd & point) const;

  /// Retrieve the parent node of the function approximator used at this point
  const FunctionApproximator &
  getPreLeafApproximator(const Eigen::VectorXd & point) const;

  /// Copy current FATree and replace the functionApproximator at 'point' by fa,
  /// then return the resulting FATree
  std::unique_ptr<FATree> copyAndReplaceLeaf(const Eigen::VectorXd & point,
                                             std::unique_ptr<FunctionApproximator> fa) const;

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
