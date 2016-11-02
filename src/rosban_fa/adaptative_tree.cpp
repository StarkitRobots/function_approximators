#include "rosban_fa/adaptative_tree.h"

#include "rosban_regression_forests/tools/statistics.h"

AdaptativeTree::AdaptiveTree()
  : nb_samples(100),
    cv_ratio(1.0)
{
}

AdaptativeTree::~AdaptativeTree() {}

std::unique_ptr<FunctionApproximator>
AdaptativeTree::train(RewardFunction rf, std::default_random_engine * engine)
{
  std::unique_ptr<FunctionApproximator> result;
  // Several generations are used
  for (int generation = 0; generation < nb_generations; generation++)
  {
    initTree(engine);
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
  working_tree.reset(new FATree());
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

std::unique_ptr<FunctionApproximator>
AdaptativeTree::buildApproximator(ApproximatorCandidate & candidate,
                                  std::default_random_engine * engine)
{
  // Initializing variables
  double best_loss = candidate.loss;
  std::vector<ApproximatorCandidate> childs;
  // Getting splits
  std::vector<std::unique_ptr<Split>> split_candidates = getSplitCandidates(samples);
  // Checking if a split improves loss criteria
  for (size_t split_idx = 0; split_idx < split_candidates.size(); split_idx++)
  {
    // Stored data for evaluation of the split
    std::vector<Eigen::MatrixXd> spaces, samples;
    std::vector<double> losses;
    std::vector<std::unique_ptr<FunctionApproximator> function_approximators;
    // Separate elements and space with loss
    int nb_elements = split_candidates[split_idx]->getNbElements();
    samples = split_candidates[split_idx]->splitEntries(candidate.parameters_set);
    spaces = split_candidates[split_idx]->splitSpace(candidate.parameters_space);
    // Estimate losses and function approximators for all elements of the split
    double total_loss = 0;
    double total_weight = 0;
    for (int elem_id = 0; elem_id < nb_elements; elem_id++)
    {
      // Computing values
      std::unique_ptr<FunctionApproximator> fa = optimizeAction(samples[elem_id]);
      Eigen::MatrixXd cv_set = getCrossValidationSet(samples[elem_id], spaces[elem_id]);
      double loss = computeLoss(fa, cv_set, engine);
      // Storing values
      function_approximators.push_back(std::move(fa));
      losses.push_back(loss);
      // Updating values
      // TODO: validate weighting method
      double weight = sampes[elem_id].size();
      total_loss += loss * sampes[elem_id].size();
      total_weight += weight;
    }
    double avg_split_loss = total_loss / total_weight;
    // If split is currently the best, store its internal data
    if (avg_split_loss < best_loss)
    {
      best_loss = avg_split_loss;
      childs.clear();
      childs.resize(nb_elements);
      for (int elem_id = 0; elem_id < nb_elements; elem_id++)
      {
        childs[elem_id].approximator = std::move(function_approximators[elem_id]);
        childs[elem_id].parameters_set = samples[elem_id];
        childs[elem_id].parameters_space = spaces[elem_id];
        childs[elem_id].loss = losses[elem_id];
      }
    }
  }
  // If no interesting split has been fonund
  if (childs.size() == 0)
  {
    ProcessedLeaf leaf;
    leaf.space = candidate.parameters_space;
    leaf.loss = best_loss;
    leaf.nb_samples = candidate.parameters_set;
    processed_leaves.push_back(leaf);
  }
  else
  {
    //TODO: build a tree
  }
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

std::vector<std::unique_ptr<Split>>
AdaptativeTree::getSplitCandidates(const Eigen::MatrixXd & samples)
{
  std::vector<std::unique_ptr<Split>> result;
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
      result.push_back(std::unique_ptr<Split>(new OrthogonalSplit(dim, split_value)));
    }
  }
  return result;
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
