#include "rosban_fa/adaptative_tree.h"

#include "rosban_regression_forests/tools/statistics.h"

AdaptativeTree::AdaptiveTree()
  : nb_samples(100),
    cv_ratio(1.0)
{
}

AdaptativeTree::~AdaptativeTree() {}

std::unique_ptr<FunctionApproximator> AdaptativeTree::train(RewardFunction rf)
{
  std::unique_ptr<FunctionApproximator> result;
  // Several generations are used
  for (int generation = 0; generation < nb_generations; generation++)
  {
    initTree();
    while(!pending_leaves.empty())
    {
      PendingLeaf leaf = pending_leaves.front();
      pending_leaves.pop_front();
      treatLeaf(leaf);
    }
  }
}

void AdaptativeTree::updateSamples(std::default_random_engine * engine)
{
  // On first generation get samples from random
  if (processed_leaves.empty()) {
    parameters_set = rosban_random::getUniformSamples(parameter_limits,
                                                      nb_samples,
                                                      engine);
  }
  else
  {
    // TODO: implement function to draw a given number of samples from a weighted set
    //       eventually propose two versions:
    //       1. unordered vector
    //       2. map of indices to count of samples
    // then:
    // - Use version 2 to generate the real samples and store them
    throw std::logic_error("AdaptativeTree::updateSamples: unimplemented part");
  }
}

void AdaptativeTree::initTree(std::default_random_engine * engine)
{
  // Checking if context is appropriate
  if (working_tree)
    throw std::logic_error("AdaptativeTree::initTree: working_tree was not null");
  if (pending_leaves.size() != 0)
    throw std::logic_error("AdaptativeTree::initTree: pending_leaves was not empty");
  // Initialize working_tree
  working_tree.reset(new regression_forests::Tree());
  working_tree->root = new regression_forests::Node();
  // Setting up root_leaf
  PendingLeaf root_leaf;
  root_leaf.node = working_tree->root;
  root_leaf.space = parameters_limits;
  root_leaf.parameters_set = generateParametersSet(engine);
  for (int id = 0; id < parameters_set.size(); i++) {
    root_leaf.indices[id] = id;
  }
  pending_leaves.push_front(root_leaf, engine);
}


void AdaptativeTree::getSamplesMatrix(const std::vector<int> & indices)
{
  if (indices.size() == 0)
    throw std::logic_error("AdaptativeTree::getSamplesMatrix: indices is empty");
  Eigen::MatrixXd result(getParametersDim(), indices.size());
  for (int r_idx = 0; r_idx < indices.size(); r_idx++)
  {
    result.col(r_idx) = parameters_set[indices[r_idx]];
  }
  return result;
}

std::vector<regression_forests::OrthogonalSplit>
AdaptativeTree::getSplitCandidates(const Eigen::MatrixXd & samples)
{
  std::vector<regression_forests::OrthogonalSplit> result;
  /// No sense to build up quartiles if there is less than 4 samples
  if (samples.cols() < 4) return result;
  /// Compute quartiles along every dimension
  for (int dim = 0; dim < getParametersDim(); dim++)
  {
    std::vector<double> values;
    for (int i = 0; i < samples.cols(); i++)
    {
      values.push_back(samples(dim,i));
    }
    /// TODO: handle cases with repeated values
    std::vector<double> split_values = regression_forests::Statistics::getQuartiles(values);
    for (double split_value : split_values)
    {
      result.push_back(OrthogonalSplit(dim, split_value));
    }
  }
  return result;
}

void AdaptativeTree::treatLeaf(PendingLeaf & leaf, std::default_random_engine * engine)
{
  // Initialize properties
  double best_loss = leaf.loss;
  regression_forests::OrthogonalSplit * best_split = nullptr;
  // Evaluate all splits can be used ?
  for (const regression_forests::OrthogonalSplit & split : getSplitCandidates())
  {
    // Getting samples collections
    Eigen::MatrixXd lower_samples, upper_samples;
    split.splitEntries(samples, &lower_samples, &upper_samples);
    // Getting spaces
    Eigen::MatrixXd lower_space, upper_space;
    split.splitSpace(leaf.space, lower_space, upper_space);
    // Optimizing constant action for each space
    // TODO: handle more methods with genericity
    // - Ideally, optimizeAction should return a function approximator
    // TODO: Use code factorization
    Eigen::VectorXd lower_action = optimizeAction(lower_samples, lower_space, engine);
    Eigen::VectorXd lower_cv_set = getCrossValidationSet(lower_space, lower_samples.size());
    double lower_loss = computeLoss(lower_action, lower_space, engine);
    Eigen::VectorXd upper_action = optimizeAction(upper_samples, upper_space, engine);
    Eigen::VectorXd upper_cv_set = getCrossValidationSet(upper_space, upper_samples.size());
    double upper_loss = computeLoss(upper_action, upper_space, engine);
    // Establishing weights
    // TODO: Validate this way of weighting
    double lower_weight = lower_samples.size();
    double upper_weight = upper_samples.size();
    double split_loss;
    split_loss = lower_loss * lower_weight + upper_loss * upper_weight;
    split_loss /= (lower_weight + upper_weight);//Normalization
    // If this split better than those previously met : remember it
    if (split_loss < best_loss)
    {
      Eigen::VectorXd best_action
    }
  }
}

Eigen::MatrixXd AdaptativeTree::getCrossValidationTest(const Eigen::MatrixXd & space,
                                                       int training_set_size,
                                                       std::default_random_engine * engine)
{
  double cv_set_size = training_set_size * cv_ration;
  return rosban_utils::getUniformSamplesMatrix(space, cv_set_size, engine);
}

double AdaptativeTree::computeLoss()
{
  ///TODO
}
