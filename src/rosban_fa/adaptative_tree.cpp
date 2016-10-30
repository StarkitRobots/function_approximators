#include "rosban_fa/adaptative_tree.h"

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
    // 1. Update currently used samples
    updateSamples();
    // 2. Create a tree and add it's root to Pending
    initTree();
    
  }
}

void AdaptativeTree::updateSamples(std::default_random_engine * engine)
{
  // On first generation get samples from random
  if (processed_leaves.size() == 0) {
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

void AdaptativeTree::initTree()
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
  root_leaf.indices.resize(parameters_set.size());
  for (int id = 0; id < parameters_set.size(); i++) {
    root_leaf.indices[id] = id;
  }
  pending_leaves.push_front(root_leaf);
}
