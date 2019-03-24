#pragma once

#include "rhoban_fa/function_approximator.h"
#include "rhoban_fa/split.h"

#include <memory>

namespace rhoban_fa
{
/// This class is a FunctionApproximator hub, in which every child is a
/// FunctionApproximator itself. It is mainly used to represent trees of
/// FunctionApproximator. Each child of the hub is used to approximate a
/// subset of the input space
///
/// Each node stores the numbers of nodes inside each child, this can be used
/// to access a node of the tree by only providing its ID or to retrieve the ID
/// of a node belonging to a given point.
///
/// Modification of the tree structure require to update the node counter:
/// - see: updateNodesCount
class FATree : public FunctionApproximator
{
public:
  FATree();
  /// Transmit ownership and clear vector 'children'
  FATree(std::unique_ptr<Split> split, std::vector<std::unique_ptr<FunctionApproximator>>& children);
  virtual ~FATree();

  const Split& getSplit() const;

  virtual std::unique_ptr<FunctionApproximator> clone() const override;

  int getOutputDim() const override;

  /// Add all the leaf spaces to the given vector
  void addSpaces(const Eigen::MatrixXd& global_space, std::vector<Eigen::MatrixXd>* spaces) const;

  /// Retrieve the elemental function approximator used at this point
  const FunctionApproximator& getLeafApproximator(const Eigen::VectorXd& point) const;

  /// Retrieve the first parent node of the function approximator used at this point
  const FATree& getPreLeafApproximator(const Eigen::VectorXd& point) const;

  /// Warning, since the structure is changed, it is mandatory to use the
  /// updateNodesCount function after replacing an approximator before using
  /// other methods based on node_id
  void replaceApproximator(const Eigen::VectorXd& point, std::unique_ptr<FunctionApproximator> fa);

  /// Copy current FATree and replace the functionApproximator at 'point' by fa,
  /// then return the resulting FATree
  std::unique_ptr<FATree> copyAndReplaceLeaf(const Eigen::VectorXd& point,
                                             std::unique_ptr<FunctionApproximator> fa) const;

  // Due to polymorphism + override of a function named 'predict', we need to
  // specifically mention the functions from FunctionApproximator
  using FunctionApproximator::predict;

  virtual void predict(const Eigen::VectorXd& input, Eigen::VectorXd& mean, Eigen::MatrixXd& covar) const override;

  virtual void gradient(const Eigen::VectorXd& input, Eigen::VectorXd& gradient) const override;

  virtual void getMaximum(const Eigen::MatrixXd& limits, Eigen::VectorXd& input, double& output) const override;

  /// How many nodes contains the tree (require a call to updateNodesCount since
  /// last modification)
  int getNodesCount() const;

  /// Modify children_sizes to ensure they contain the appropriate number of nodes
  void updateNodesCount();

  /// Return the nodeIds of all the leaves, should only be called on the root of the tree
  /// see updateNodesCount
  std::vector<int> getLeavesId() const;

  /// Return the nodeId of the leaf corresponding to the provided point
  /// see updateNodesCount
  int getLeafId(const Eigen::VectorXd& point) const;

  virtual int getClassID() const override;
  virtual int writeInternal(std::ostream& out) const override;
  virtual int read(std::istream& in) override;

  virtual std::string toString() const override;

protected:
  /// Push the ids of all the leaves in the tree inside the provided vector
  /// see updateNodesCount
  void fillLeavesId(std::vector<int>* leaves_id, int offset) const;

  /// Throws an explicit exception if content was not properly set
  void checkConsistency(const std::string& caller_name) const;

  /// The input space is separated in several parts (eventually more than two)
  std::unique_ptr<Split> split;
  /// List of the children which can be used for the tree
  std::vector<std::unique_ptr<FunctionApproximator>> children;

  /// How many nodes in children (child included)
  std::vector<int> children_sizes;
};

}  // namespace rhoban_fa
