#pragma once

#include "rosban_fa/optimizer_trainer.h"
#include "rosban_fa/split.h"

#include "rosban_bbo/optimizer.h"

#include <deque>

namespace rosban_fa
{

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
class AdaptativeTree : public OptimizerTrainer
{
public:
  /// Candidate for approximations are composed of:
  /// - approximator: The candidate for approximating this function
  /// - parameters_set: The set of parameters used for training
  /// - parameters_space: Which space is concerned by this approximator (used for cross-validation)
  /// - reward: The expected reward for the Function Approximator over the given parameters_space
  struct ApproximatorCandidate
  {
    std::unique_ptr<FunctionApproximator> approximator;
    Eigen::MatrixXd parameters_set;
    Eigen::MatrixXd parameters_space;
    double reward;
    int depth;
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


  typedef std::function<double(const FunctionApproximator & policy,
                               std::default_random_engine * engine)> EvaluationFunction;
  AdaptativeTree();
  virtual ~AdaptativeTree();

  virtual std::unique_ptr<FunctionApproximator>
  train(RewardFunction rf, std::default_random_engine * engine);

  /// Reset all internal memory of the algorithm
  virtual void reset() override;

  /// Generate a new set of samples:
  /// - Using uniformous random for first generation
  /// - Based on processed leaves on further generation
  Eigen::MatrixXd generateParametersSet(std::default_random_engine * engine);

  /// Initialize the tree for the provide set of samples
  /// - generate the set of samples to be used for the generation
  /// - use recursion to determine the complete FunctionApproximator
  std::unique_ptr<FunctionApproximator> runGeneration(RewardFunction rf,
                                                      std::default_random_engine * engine);

  /// Use an initial guess with its own value and if splitting improves, then split it
  /// - Add leafs created to processed leafs
  /// - Return the function approximator chosen by candidate
  /// - Might consume candidate.approximator
  std::unique_ptr<FunctionApproximator> buildApproximator(RewardFunction rf,
                                                          ApproximatorCandidate & candidate,
                                                          std::default_random_engine * engine);

  /// Compute the split candidates from a Matrix in which:
  /// - Lines are dimensions
  /// - Each column is one of the samples
  std::vector<std::unique_ptr<Split>> getSplitCandidates(const Eigen::MatrixXd & samples);

  /// Build a cross-validation state for the given space and the given training_set_size
  Eigen::MatrixXd getCrossValidationSet(const Eigen::MatrixXd & space,
                                        int training_set_size,
                                        std::default_random_engine * engine);

  /// Update the candidate by training a model for its training state
  /// and training space
  void updateAction(RewardFunction rf,
                    ApproximatorCandidate & candidate,
                    const Eigen::VectorXd & guess,
                    std::default_random_engine * engine);

  /// Return an optimized constant policy
  /// guess is the action currently guessed to be optimal
  std::unique_ptr<FunctionApproximator>
  optimizeConstantPolicy(EvaluationFunction policy_evaluator,
                         const Eigen::VectorXd & guess,
                         std::default_random_engine * engine);
    
  /// Return an optimized linear policy
  /// guess is the action currently guessed to be optimal
  std::unique_ptr<FunctionApproximator>
  optimizeLinearPolicy(EvaluationFunction policy_evaluator,
                       const Eigen::MatrixXd & parameters_space,
                       const Eigen::VectorXd & guess,
                       std::default_random_engine * engine);

  /// Update the 'reward' field of the candidate, using his 'approximator'
  /// and cross-validation
  void updateReward(RewardFunction rf,
                    ApproximatorCandidate & candidate,
                    std::default_random_engine * engine);

  /// Compute the average reward for the given function approximator
  double computeAverageReward(RewardFunction rf,
                              const FunctionApproximator & fa,
                              const Eigen::VectorXd & parameters,
                              std::default_random_engine * engine);

  /// Return a function
  EvaluationFunction getEvaluationFunction(RewardFunction rf,
                                           const Eigen::MatrixXd & training_set);

  virtual std::string class_name() const;
  virtual void to_xml(std::ostream &out) const override;
  virtual void from_xml(TiXmlNode *node) override;

private:

  static void print(const ApproximatorCandidate & candidate, std::ostream & out);


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

  /// Number of evaluations required to estimate the average reward for a set:
  /// (parameter, action)
  int evaluation_trials;

  /// The optimizer used to train parameters of the models
  std::unique_ptr<rosban_bbo::Optimizer> model_optimizer;

  /// Variable stored to make debugging easier
  int nb_samples_treated;

  /// How much talkative is the process ?
  /// 0: Silent
  /// 1: Progress (Print a header for each generation and number of leafs treated)
  /// 2: Results  (Print a summary of each chosen candidate)
  /// 3: Detailed (Print a header for split candidates and their associated scores)
  /// TODO: create a 4th level (would probably help)
  int verbosity;

  /// Are the training samples generated once for every generation (true) or are
  /// they renewed for each split test?
  bool reuse_samples;

  /// Are points splits allowed (2^n child with n= parametersDim)
  bool use_point_splits;

  /// Maximal depth (negative value means infinite)
  int max_depth;

  /// Amplitude of linear coefficients:
  /// true : For each hyperrectangle, it is possible to have the minimal action
  ///        at a corner and the maximal action at another
  /// false: For each hyperrectangle, it is possible to have the minimal action
  ///        at a 'edge' and the maximal at the opposite 'edge' (only one
  ///        dimension changed)
  bool narrow_linear_slope;

  /// When constant model is trained, does it uses guess if provided
  bool constant_uses_guess;

  /// When linear model is trained, does it uses guess if provided
  bool linear_uses_guess;
};

}
