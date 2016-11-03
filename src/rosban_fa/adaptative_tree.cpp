#include "rosban_fa/adaptative_tree.h"

#include "rosban_regression_forests/tools/statistics.h"

namespace rosban_fa
{

AdaptativeTree::AdaptiveTree()
  : nb_generations(1),
    nb_samples(100),
    cv_ratio(1.0),
    evaluation_trials(10)
{
}

AdaptativeTree::~AdaptativeTree() {}

std::unique_ptr<FunctionApproximator>
AdaptativeTree::train(RewardFunction rf, std::default_random_engine * engine)
{
  std::unique_ptr<FunctionApproximator> result;
  for (int generation = 0; generation < nb_generations; generation++)
  {
    result =  runGeneration(engine);
  }
  return result;
}

void AdaptativeTree::generateParametersSet(std::default_random_engine * engine)
{
  Eigen::MatrixXd parameters_set;
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
  return parameters_set;
}

std::unique_ptr<FunctionApproximator>
AdaptativeTree::runGeneration(std::default_random_engine * engine)
{
  updateSamples(engine);
  // Setting up first candidate
  ApproximatorCandidate candidate;
  candidate.parameters_set = generateParametersSet();
  candidate.parameters_space = parameters_limits;
  candidate.approximator = optimizeAction(candidate.parameters_set,
                                          candidate.parameters_space,
                                          engine);
  updateReward(candidate);
  return buildApproximator(candidate, engine);
}

std::unique_ptr<FunctionApproximator>
AdaptativeTree::buildApproximator(ApproximatorCandidate & candidate,
                                  std::default_random_engine * engine)
{
  // Initializing variables
  double best_reward = candidate.reward;
  std::unique_ptr<Split> best_split;
  std::vector<ApproximatorCandidate> childs;
  // Getting splits
  std::vector<std::unique_ptr<Split>> split_candidates = getSplitCandidates(samples);
  // Checking if a split improves reward criteria
  for (size_t split_idx = 0; split_idx < split_candidates.size(); split_idx++)
  {
    // Stored data for evaluation of the split
    std::vector<Eigen::MatrixXd> spaces, samples;
    std::vector<double> rewards;
    std::vector<std::unique_ptr<FunctionApproximator> function_approximators;
    // Separate elements and space with reward
    int nb_elements = split_candidates[split_idx]->getNbElements();
    samples = split_candidates[split_idx]->splitEntries(candidate.parameters_set);
    spaces = split_candidates[split_idx]->splitSpace(candidate.parameters_space);
    // Estimate rewards and function approximators for all elements of the split
    double total_reward = 0;
    double total_weight = 0;
    for (int elem_id = 0; elem_id < nb_elements; elem_id++)
    {
      // Computing values
      std::unique_ptr<FunctionApproximator> fa = optimizeAction(samples[elem_id]);
      double reward = computeReward(fa, cv_set, engine);
      // Storing values
      function_approximators.push_back(std::move(fa));
      rewards.push_back(reward);
      // Updating values
      // TODO: validate weighting method
      double weight = sampes[elem_id].size();
      total_reward += reward * sampes[elem_id].size();
      total_weight += weight;
    }
    double avg_split_reward = total_reward / total_weight;
    // If split is currently the best, store its internal data
    if (avg_split_reward > best_reward)
    {
      best_reward = avg_split_reward;
      childs.clear();
      childs.resize(nb_elements);
      for (int elem_id = 0; elem_id < nb_elements; elem_id++)
      {
        childs[elem_id].approximator = std::move(function_approximators[elem_id]);
        childs[elem_id].parameters_set = samples[elem_id];
        childs[elem_id].parameters_space = spaces[elem_id];
        childs[elem_id].reward = rewards[elem_id];
      }
    }
  }
  // If no interesting split has been fonund
  if (childs.size() == 0)
  {
    ProcessedLeaf leaf;
    double custom_reward = 0;//TODO
    leaf.space = candidate.parameters_space;
    leaf.loss = candidate.reward - custom_reward;
    leaf.nb_samples = candidate.parameters_set.size();
    processed_leaves.push_back(leaf);
    return std::move(candidate.approximator);
  }
  else
  {
    std::vector<std::unique_ptr<FunctionApproximator>> childs_fa;
    for (size_t child_id = 0; child_id < childs.size(); child_id++)
    {
      childs_fa.push_back(std::move(childs[child_id].approximator));
    }
    return std::unique_ptr<Tree>(new Tree(best_split, childs_fa));
  }
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

void AdaptativeTree::updateReward(ApproximatorCandidate & candidate,
                                std::default_random_engine * engine)
{
  Eigen::MatrixXd cv_set = getCrossValidationSet(candidate.parameters_space,
                                                 candidate.parameters_set.size(),
                                                 engine);
  double total_reward = 0;
  for (int cv_idx = 0; cv_idx < cv_set.cols(); cv_idx++)
  {
    total_reward += computeAverageReward(rf,
                                         candidate.approximator,
                                         cv_set.col(cv_idx),
                                         engine);
  }
  candidate.reward = total_reward / cv_set.cols();
}

double FunctionApproximator::computeAverageReward(RewardFunction rf,
                                                  const FunctionApproximator & fa,
                                                  const Eigen::VectorXd & parameters,
                                                  std::default_random_engine * engine)
{
  double total_reward = 0;
  for (int trial = 0; trial < evaluation_trials; trial++)
  {
    total_reward += rf(parameters, fa_action, engine);
  }
  return total_reward / evaluation_trials;
}

EvaluationFunction
FunctionApproximator::getEvaluationFunction(RewardFunction rf,
                                            const Eigen::MatrixXd & training_set)
{
  return
    [training_set, action_limits, rf] (const FunctionApproximator & action,
                                       std::default_random_engine * engine)
    {
      double reward = 0;
      for (int col = 0; col < training_set.cols(); col++)
      {
        Eigen::VectorXd parameters = training_set.col(col);
        Eigen::VectorXd action;
        Eigen::MatrixXd covar;
        policy.predict(parameters, action, covar);
        reward += rf(parameters, action, engine);
      }
      return reward / training_set.cols();
    }
}

//TODO replace by function optimizer
std::unique_ptr<FunctionApproximator>
FunctionApproximator::optimizeAction(RewardFunction rf,
                                     const Eigen::MatrixXd & training_set,
                                     const Eigen::MatrixXd & action_limits,
                                     std::default_random_engine * engine)
{
  std::function<double(const Eigen::MatrixXd &)> 
}

}
