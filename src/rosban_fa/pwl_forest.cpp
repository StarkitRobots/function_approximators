#include "rosban_fa/pwl_forest.h"

#include "rosban_regression_forests/algorithms/extra_trees.h"
#include "rosban_regression_forests/approximations/approximation_type.h"
#include "rosban_regression_forests/approximations/pwl_approximation.h"

using regression_forests::ApproximationType;
using regression_forests::PWLApproximation;
using regression_forests::ExtraTrees;
using regression_forests::TrainingSet;
using regression_forests::Forest;
using regression_forests::Tree;
using regression_forests::Node;

namespace rosban_fa
{

PWLForest::PWLForest() {}

PWLForest::PWLForest(std::unique_ptr<Forests> forests_,
                     int max_action_tiles_)
  : forests(std::move(forests_)), max_action_tiles(max_action_tiles_)
{}

PWLForest::~PWLForest() {}

int PWLForest::getOutputDim() const
{
  if (!forests) return 0;
  return forests->size();
}

void PWLForest::predict(const Eigen::VectorXd & input,
                        Eigen::VectorXd & mean,
                        Eigen::MatrixXd & covar) const
{
  int O = getOutputDim();
  mean = Eigen::VectorXd::Zero(O);
  covar = Eigen::MatrixXd::Zero(O,O);
  for (int output_dim = 0; output_dim < O; output_dim++)
  {
    mean(output_dim) = (*forests)[output_dim]->getValue(input);
    covar(output_dim, output_dim) = (*forests)[output_dim]->getVar(input);
  }
}

void PWLForest::gradient(const Eigen::VectorXd & input,
                         Eigen::VectorXd & gradient) const
{
  (void) input;
  (void) gradient;
  throw std::runtime_error("PWLForest::gradients: not implemented");
}

void PWLForest::getMaximum(const Eigen::MatrixXd & limits,
                           Eigen::VectorXd & input,
                           double & output) const
{
  check1DOutput("getMaximum");
  //TODO: optional other way to compute max (gradient based)
  std::unique_ptr<regression_forests::Tree> sub_tree;
  sub_tree = (*forests)[0]->unifiedProjectedTree(limits, max_action_tiles);

  std::pair<double, Eigen::VectorXd> max_pair;
  max_pair = sub_tree->getMaxPair(limits);

  input = max_pair.second;
  output = max_pair.first;
}

int PWLForest::getClassID() const
{
  return FunctionApproximator::PWLForest;
}

int PWLForest::writeInternal(std::ostream & out) const
{
  (void) out;
  throw std::logic_error("PWLForest::writeInternal:not implemented");
}

int PWLForest::read(std::istream & in)
{
  (void) in;
  throw std::logic_error("PWLForest::read: not implemented");
}

}
