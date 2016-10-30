#pragma once

#include "rosban_fa/optimizer_trainer.h"

/// This classes uses the following concepts:
/// - A set of samples is used to build tree-based approximations
/// - The set of samples can only be splitted on medians and is always splitted
///   on the best median
/// - Evaluation of interpolation methods is performed through cross-validation,
///   in order to avoid over-fitting (on generated states)
/// - At each step, multiple models are fitted and the best one is chosen
///   (according to cross-validation) 
/// - Multiple generations can be used, in this case each generation is based on
///   the previous one and has more chance to optimize samples in areas where
///   the loss due to use of the approximator is expected to be the highest
class AdaptativeTree : public rosban_fa::OptimizerTrainer
{
public:

  AdaptativeTree();
  virtual ~AdaptativeTree();

  virtual std::unique_ptr<FunctionApproximator>
  train(RewardFunction rf);

  /// Generate a new set of samples:
  /// - Using uniformous random if no previous samples existed
  /// - Based on previous samples and splits if samples have been processed
  void updateSamples(std::default_random_engine * engine);

  /// Initialize the tree for the provide set of samples
  /// - update working_tree
  /// - add an initialized PendingLeaf to pending_leaves
  void initTree();

private:
  /// For leafs in process it is mandatory to keep track of the following information:
  /// - node: Link to the node in order to grow the tree
  /// - indices: The indices of parameters used to for the current node
  /// - space: The hyperrectangle used for current node
  struct PendingLeaf
  {
    regression_forests::Node * node;
    std::vector<int> indices;
    Eigen::MatrixXd space;
  };

  /// For processed leafs, some informations are required
  /// - space: Which was the leaf space (used to draw samples of the next generation
  /// - loss: For each leaf, a loss has been estimated using cross-validation set
  /// - nb_samples: what was the count of samples used for learning in the space
  struct ProcessedLeaf
  {
    Eigen::MatrixXd space;
    double loss;
    int nb_samples;
  };

  /// Current tree bieng built:
  std::unique_ptr<regression_forests::Tree> working_tree;

  /// Leafs currently being processed
  std::deque<PendingLeaf> pending_leaves;

  /// Leafs which have already been processed
  std::deque<ProcessedLeaf> processed_leaves;

  /// The current set of parameters used to grow the tree
  std::vector<Eigen::VectorXd> parameters_set;

  /// Number of generations used for training
  int nb_generations;

  /// Numbers of samples wished
  int nb_samples;

  /// TODO: handle nb_samples growth

  /// Cross-Validation set size ratio of samples used for cross-validation
  /// Number of samples used for cross_validation is nb samples used for
  /// training times cv_ratio
  double cv_ratio;
};
