#pragma once

#include "rosban_fa/optimizer_trainer.h"

/// TODO: regression_forests::Tree are not appropriate, because they represent a function
///       from R^n to R. Thus, a specific class has to be designed to handle trees of FA.
///       Those trees could also use another split interface, more generic and more
///       convenient (supporting multiple childs, returning vector of spaces/samples)
///       this type of interface can support easily multiple splits such as grid split
///       non-orthogonal splits are more complex since the resulting spaces are not
///       hyperrectangles.

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
  /// - Using uniformous random for first generation
  /// - Based on processed leaves on further generation
  void generateParametersSet(std::default_random_engine * engine);

  /// Initialize the tree for the provide set of samples
  /// - update working_tree
  /// - add an initialized PendingLeaf to pending_leaves
  void initTree(std::default_random_engine * engine);

  /// Treat the pending leaf provided:
  /// - Test multiple split options
  ///   - If a split improves the cross-validation
  ///     - Splits the node and add the two created leaves to the pending_leaves
  ///   - If no split has been found improving cross-validation
  ///     - Does not add any leaves to pending_leaves
  void treatLeaf(PendingLeaf & leaf, std::default_random_engine * engine);

  /// Build a matrix containing the samples at the provided indices
  void getSamplesMatrix(const std::vector<int> & indices);

  /// Compute the split candidates from a Matrix in which:
  /// - Lines are dimensions
  /// - Each column is one of the samples
  std::vector<regression_forests::OrthogonalSplit>
  getSplitCandidates(const Eigen::MatrixXd & samples);

  /// Build a cross-validation state for the given space and the given training_set_size
  Eigen::MatrixXd getCrossValidationSet(const Eigen::MatrixXd & space,
                                        int training_set_size,
                                        std::default_random_engine * engine);

  /// TODO:
  double optimizeAction(const Eigen::MatrixXd & parameters_set,
                        const Eigen::MatrixXd & space,
                        std::default_random_engine * engine);

  /// TODO:
  /// - Require a little bit of thinking
  /// - Eventually implement different options (squared_loss, etc)
  double computeLoss();

private:
  /// For leafs in process it is mandatory to keep track of the following information:
  /// - node: Link to the node in order to grow the tree
  /// - parameters_set: the set of parameters used for this leaf
  /// - space: The hyperrectangle used for current node
  /// - loss:
  struct PendingLeaf
  {
    regression_forests::Node * node;
    Eigen::MatrixXd parameters_set;
    Eigen::MatrixXd space;
    double loss;
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
