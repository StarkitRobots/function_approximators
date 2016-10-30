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

private:
  /// Current samples used
  std::vector<Eigen::VectorXd> samples;
};
