#include "rosban_fa/adaptative_tree.h"

#include "rosban_fa/constant_approximator.h"
#include "rosban_fa/fa_tree.h"
#include "rosban_fa/linear_approximator.h"
#include "rosban_fa/orthogonal_split.h"
#include "rosban_fa/point_split.h"

#include "rosban_regression_forests/tools/statistics.h"

#include "rosban_bbo/optimizer_factory.h"

#include "rosban_random/tools.h"

#include "rosban_utils/multi_core.h"

namespace rosban_fa
{

AdaptativeTree::AdaptativeTree()
  : nb_generations(1),
    nb_samples(100),
    cv_ratio(1.0),
    evaluation_trials(10),
    nb_samples_treated(0)
{
}

AdaptativeTree::~AdaptativeTree() {}

std::unique_ptr<FunctionApproximator>
AdaptativeTree::train(RewardFunction rf, std::default_random_engine * engine)
{
  std::unique_ptr<FunctionApproximator> result;
  for (int generation = 1; generation <= nb_generations; generation++)
  {
    std::cout << "Generation " << generation << "/" << nb_generations << std::endl;
    result =  runGeneration(rf, engine);
  }
  return result;
}

void AdaptativeTree::reset() {
  processed_leaves.clear();
}

Eigen::MatrixXd AdaptativeTree::generateParametersSet(std::default_random_engine * engine)
{
  Eigen::MatrixXd parameters_set;
  // On first generation get samples from random
  if (processed_leaves.empty()) {
    parameters_set = rosban_random::getUniformSamplesMatrix(parameters_limits,
                                                            nb_samples,
                                                            engine);
  }
  else
  {
    // Computing weights
    int nb_leaves = processed_leaves.size();
    std::vector<double> weights(nb_leaves);
    for (int i = 0; i < nb_leaves; i++)
    {
      weights[i] = 1.0;
    }
    // space_index -> nb_samples wished
    std::map<int,int> space_occurences;
    space_occurences = rosban_random::sampleWeightedIndicesMap(weights,
                                                               nb_samples,
                                                               engine);
    // Computing parameters set
    int param_dims = parameters_limits.rows();
    parameters_set = Eigen::MatrixXd(param_dims, nb_samples);
    int sample_id = 0;
    for (size_t leaf_id = 0; leaf_id < processed_leaves.size(); leaf_id++) {
      const ProcessedLeaf & leaf = processed_leaves[leaf_id];
      int leaf_nb_samples = space_occurences[leaf_id];
      // Drawing samples for this leaf
      Eigen::MatrixXd leaf_samples;
      leaf_samples = rosban_random::getUniformSamplesMatrix(leaf.space,
                                                            leaf_nb_samples,
                                                            engine);
//      std::cout << "Placing " << leaf_nb_samples << " samples from "
//                << sample_id << std::endl;
      // Affecting samples
      parameters_set.block(0, sample_id,
                           param_dims, leaf_nb_samples) = leaf_samples;
      sample_id += leaf_nb_samples;
    }
  }
//  std::cout << "Parameters set: " << std::endl
//            << parameters_set.transpose() << std::endl;
  return parameters_set;
}

std::unique_ptr<FunctionApproximator>
AdaptativeTree::runGeneration(RewardFunction rf,
                              std::default_random_engine * engine)
{
  // Reset number of samples trained, valid only for generation
  nb_samples_treated = 0;
  // Setting up first candidate
  ApproximatorCandidate candidate;
  candidate.parameters_set = generateParametersSet(engine);
  candidate.parameters_space = parameters_limits;
  candidate.approximator = optimizeAction(rf,
                                          candidate.parameters_set,
                                          candidate.parameters_space,
                                          engine);
  updateReward(rf, candidate, engine);
  return buildApproximator(rf, candidate, engine);
}

std::unique_ptr<FunctionApproximator>
AdaptativeTree::buildApproximator(RewardFunction rf,
                                  ApproximatorCandidate & candidate,
                                  std::default_random_engine * engine)
{
  // Initializing variables
  double best_reward = candidate.reward;
  std::unique_ptr<Split> best_split;
  std::vector<ApproximatorCandidate> childs;
  // Getting splits
  std::vector<std::unique_ptr<Split>> split_candidates;
  split_candidates = getSplitCandidates(candidate.parameters_set);

  // Debug time:
//  std::cout << "### buildApproximator ###" << std::endl
//            << "space:" << std::endl
//            << candidate.parameters_space << std::endl
//            << "nb_samples: " << candidate.parameters_set.cols() << std::endl;

  // Checking if a split improves reward criteria
  for (size_t split_idx = 0; split_idx < split_candidates.size(); split_idx++)
  {
    std::unique_ptr<Split> current_split = std::move(split_candidates[split_idx]);
//    std::cout << "\tSplit: " << current_split->toString() << std::endl;
    // Stored data for evaluation of the split
    std::vector<Eigen::MatrixXd> spaces, samples;
    std::vector<double> rewards;
    std::vector<std::unique_ptr<FunctionApproximator>> function_approximators;
    // Separate elements and space with reward
    int nb_elements = current_split->getNbElements();
    samples = current_split->splitEntries(candidate.parameters_set);
    spaces = current_split->splitSpace(candidate.parameters_space);
    // If a split would result on getting one of the space empty, refuse it
    bool creates_empty_spaces = false;
    for (const Eigen::MatrixXd & samples_set : samples) {
      if (samples_set.cols() == 0) {
        creates_empty_spaces = true;
        break;
      }
    }
    if (creates_empty_spaces) {
//      std::cout << "\t-> split is creating empty spaces: canceled" << std::endl;
      continue;
    }
    // Estimate rewards and function approximators for all elements of the split
    double total_reward = 0;
    double total_weight = 0;
    for (int elem_id = 0; elem_id < nb_elements; elem_id++)
    {
      // Computing values
      ApproximatorCandidate current_candidate;
      current_candidate.parameters_set = samples[elem_id];
      current_candidate.parameters_space = spaces[elem_id];
      current_candidate.approximator = optimizeAction(rf, samples[elem_id],
                                                      spaces[elem_id], engine);
      updateReward(rf, current_candidate, engine);
      // Storing values
      function_approximators.push_back(std::move(current_candidate.approximator));
      rewards.push_back(current_candidate.reward);
      // Updating values
      // TODO: validate weighting method
      double weight = samples[elem_id].size();
      total_reward += current_candidate.reward * weight;
      total_weight += weight;
//      std::cout << "\t# Elem " << elem_id << std::endl;
//      std::cout << "\t\tSpace:" << std::endl;
//      std::cout << current_candidate.parameters_space << std::endl;
//      std::cout << "\t\tAvg Reward: " << current_candidate.reward << std::endl;
//      std::cout << "\t\tWeight: " << weight << std::endl;
    }
    double avg_split_reward = total_reward / total_weight;
    // If split is currently the best, store its internal data
    if (avg_split_reward > best_reward)
    {
//      std::cout << "\t\t<- Best split found yet" << std::endl;
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
      best_split = std::move(current_split);
    }
  }
  // If no interesting split has been fonund
  if (childs.size() == 0)
  {
    std::cout << "### A leaf has been reached" << std::endl;
    nb_samples_treated += candidate.parameters_set.cols();
    print(candidate, std::cout);
    std::cout << "Samples treated: " << nb_samples_treated << "/" 
              << nb_samples << std::endl;
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
//    std::cout << "\tAn interesting split has been found: going deeper" << std::endl;
    std::vector<std::unique_ptr<FunctionApproximator>> childs_fa;
    for (size_t child_id = 0; child_id < childs.size(); child_id++)
    {
      // Try to split the child samples if it improves the results
      childs_fa.push_back(buildApproximator(rf, childs[child_id], engine));
    }
    return std::unique_ptr<FunctionApproximator>(new FATree(std::move(best_split), childs_fa));
  }
}

std::vector<std::unique_ptr<Split>>
AdaptativeTree::getSplitCandidates(const Eigen::MatrixXd & samples)
{
  std::vector<std::unique_ptr<Split>> result;
  /// No sense to build up quartiles if there is less than 4 samples
  if (samples.cols() < 4) return result;
  /// Compute quartiles along every dimension (and storing median point)
  Eigen::VectorXd median_point = Eigen::VectorXd(getParametersDim());
  for (int dim = 0; dim < getParametersDim(); dim++)
  {
    std::vector<double> values;
    for (int i = 0; i < samples.cols(); i++)
    {
      values.push_back(samples(dim,i));
    }
    std::vector<double> split_values = regression_forests::Statistics::getQuartiles(values);
    for (double split_value : split_values)
    {
      result.push_back(std::unique_ptr<Split>(new OrthogonalSplit(dim, split_value)));
    }
    // Storing median point
    median_point(dim) = split_values[1];
  }
  // If number of points is high enough, also add the possibility to split on the median
  if (samples.cols() >= std::pow(2, getParametersDim())) {
    result.push_back(std::unique_ptr<Split>(new PointSplit(median_point)));
  }
  return result;
}

Eigen::MatrixXd AdaptativeTree::getCrossValidationSet(const Eigen::MatrixXd & space,
                                                       int training_set_size,
                                                       std::default_random_engine * engine)
{
  double cv_set_size = training_set_size * cv_ratio;
  return rosban_random::getUniformSamplesMatrix(space, cv_set_size, engine);
}

void AdaptativeTree::updateReward(RewardFunction rf,
                                  ApproximatorCandidate & candidate,
                                  std::default_random_engine * engine)
{
  Eigen::MatrixXd cv_set = getCrossValidationSet(candidate.parameters_space,
                                                 candidate.parameters_set.size(),
                                                 engine);
  double total_reward = 0;
  for (int cv_idx = 0; cv_idx < cv_set.cols(); cv_idx++)
  {
    total_reward += computeAverageReward(rf,
                                         *candidate.approximator,
                                         cv_set.col(cv_idx),
                                         engine);
  }
  candidate.reward = total_reward / cv_set.cols();
}

double AdaptativeTree::computeAverageReward(RewardFunction rf,
                                            const FunctionApproximator & fa,
                                            const Eigen::VectorXd & parameters,
                                            std::default_random_engine * engine)
{
  double total_reward = 0;
  for (int trial = 0; trial < evaluation_trials; trial++)
  {
    Eigen::VectorXd fa_action;
    Eigen::MatrixXd action_covar;
    fa.predict(parameters, fa_action, action_covar);
    total_reward += rf(parameters, fa_action, engine);
  }
  return total_reward / evaluation_trials;
}

AdaptativeTree::EvaluationFunction
AdaptativeTree::getEvaluationFunction(RewardFunction rf,
                                      const Eigen::MatrixXd & training_set)
{
  int nb_trials = evaluation_trials;
  int threads_used = std::min(nb_threads, (int)training_set.cols());
  return
    [training_set, rf, nb_trials, threads_used]
    (const FunctionApproximator & policy,
     std::default_random_engine * engine)
    {
      std::vector<double> rewards(training_set.cols());
      rosban_utils::MultiCore::StochasticTask eval_task =
        [&] (int start_idx, int end_idx, std::default_random_engine * engine)
        {
          for (int col = start_idx; col < end_idx; col++) {
            const Eigen::VectorXd & parameters = training_set.col(col);
            Eigen::VectorXd action;
            Eigen::MatrixXd covar;
            policy.predict(parameters, action, covar);
            double col_reward = 0;
            for (int trial = 0; trial < nb_trials; trial++) {
              col_reward += rf(parameters, action, engine);
            }
            rewards[col] = col_reward;
          }
        };
      // getting engines
      std::vector<std::default_random_engine> engines;
      engines = rosban_random::getRandomEngines(threads_used, engine);
      rosban_utils::MultiCore::runParallelStochasticTask(eval_task,
                                                         training_set.cols(),
                                                         &engines);
      // Summing rewards
      double total_reward = 0;
      for (double reward : rewards) {
        total_reward += reward;
      }
      return total_reward / (training_set.cols() * nb_trials);
    };
}

std::unique_ptr<FunctionApproximator>
AdaptativeTree::optimizeAction(RewardFunction rf,
                               const Eigen::MatrixXd & training_set,
                               const Eigen::MatrixXd & parameters_space,
                               std::default_random_engine * engine)
{
  EvaluationFunction policy_evaluator = getEvaluationFunction(rf, training_set);
  // Computing linear and constant policies
  std::unique_ptr<FunctionApproximator> best_constant_policy, best_linear_policy;
  best_constant_policy = optimizeConstantPolicy(policy_evaluator, engine);
  best_linear_policy = optimizeLinearPolicy(policy_evaluator, parameters_space, engine);
  // Evaluation of both policies
  double constant_score = policy_evaluator(*best_constant_policy, engine);
  double linear_score = policy_evaluator(*best_linear_policy, engine);
  // Returning the best one
  if (linear_score > constant_score)
    return std::move(best_linear_policy);
  return std::move(best_constant_policy);
}

std::unique_ptr<FunctionApproximator>
AdaptativeTree::optimizeConstantPolicy(EvaluationFunction policy_evaluator,
                                       std::default_random_engine * engine)
{
  if (!model_optimizer) {
    throw std::runtime_error("AdaptativeTree::optimizeConstantPolicy: No model optimizer available");
  }

  // Creating the reward function for constant models
  rosban_bbo::Optimizer::RewardFunc constant_model_reward_func;
  constant_model_reward_func = [policy_evaluator](const Eigen::VectorXd & parameters,
                                                  std::default_random_engine * engine)
    {
      ConstantApproximator policy(parameters);
      return policy_evaluator(policy, engine);
    };
  // Training a constant model
  model_optimizer->setLimits(actions_limits);
  Eigen::VectorXd best_action = model_optimizer->train(constant_model_reward_func, engine);
  return std::unique_ptr<FunctionApproximator>(new ConstantApproximator(best_action));
}

std::unique_ptr<FunctionApproximator>
AdaptativeTree::optimizeLinearPolicy(EvaluationFunction policy_evaluator,
                                     const Eigen::MatrixXd & parameters_space,
                                     std::default_random_engine * engine)
{
  if (!model_optimizer) {
    throw std::runtime_error("AdaptativeTree::optimizeLinearPolicy: No model optimizer available");
  }

  int parameter_dims = getParametersDim();
  int action_dims = getActionsDim();

  // Creating linear parameters space
  Eigen::MatrixXd linear_parameters_space((parameter_dims +1) * action_dims,2);
  // Bias Limits
  linear_parameters_space.block(0,0,action_dims, 2) = actions_limits;
  // Coeffs Limits
  // For each parameter, it might at most make the output vary from min to max in given space
  for (int action_dim = 0; action_dim < action_dims; action_dim++) {
    double action_amplitude = actions_limits(action_dim,1) - actions_limits(action_dim,0);
    for (int parameter_dim = 0; parameter_dim < parameter_dims; parameter_dim++) {
      double param_min = parameters_space(parameter_dim,0);
      double param_max = parameters_space(parameter_dim,1);
      // Avoiding numerical issues
      double parameter_amplitude = std::max(param_max - param_min,
                                            std::pow(10,-6));
      int index = action_dim + action_dims * (1 + parameter_dim);
      double max_coeff = action_amplitude / parameter_amplitude;
      linear_parameters_space(index, 0) = -max_coeff;
      linear_parameters_space(index, 1) =  max_coeff;
    }
  }

  // Creating the reward function for linear models
  rosban_bbo::Optimizer::RewardFunc linear_model_reward_func;
  linear_model_reward_func =
    [policy_evaluator, action_dims, parameter_dims]
    (const Eigen::VectorXd & parameters, std::default_random_engine * engine)
    {
      LinearApproximator policy(parameter_dims, action_dims, parameters);
      return policy_evaluator(policy, engine);
    };
  // Training a linear model
  model_optimizer->setLimits(linear_parameters_space);
  Eigen::VectorXd best_action = model_optimizer->train(linear_model_reward_func, engine);
  if (false)//(true) //best_action.rows() == 0)
  {
    std::cout << "Parameters space: " << std::endl
              << parameters_space << std::endl;
    std::cout << "Liner parameters space: " << std::endl
              << linear_parameters_space << std::endl;
  }
  return std::unique_ptr<FunctionApproximator>
    (new LinearApproximator(parameter_dims, action_dims, best_action));
}


std::string AdaptativeTree::class_name() const
{
  return "AdaptativeTree";
}

void AdaptativeTree::to_xml(std::ostream &out) const
{
  rosban_utils::xml_tools::write<int>   ("nb_generations"   , nb_generations   , out);
  rosban_utils::xml_tools::write<int>   ("nb_samples"       , nb_samples       , out);
  rosban_utils::xml_tools::write<int>   ("evaluation_trials", evaluation_trials, out);
  rosban_utils::xml_tools::write<double>("cv_ratio"         , cv_ratio         , out);
}

void AdaptativeTree::from_xml(TiXmlNode *node)
{
  rosban_utils::xml_tools::try_read<int>   (node, "nb_generations"   , nb_generations   );
  rosban_utils::xml_tools::try_read<int>   (node, "nb_samples"       , nb_samples       );
  rosban_utils::xml_tools::try_read<int>   (node, "evaluation_trials", evaluation_trials);
  rosban_utils::xml_tools::try_read<double>(node, "cv_ratio"         , cv_ratio         );
  rosban_bbo::OptimizerFactory().tryRead(node,"model_optimizer", model_optimizer);
}

void AdaptativeTree::print(const ApproximatorCandidate & candidate,
                           std::ostream & out) {
  out << "approximator: " << candidate.approximator->toString() << std::endl
      << "parameters_set_size: " << candidate.parameters_set.cols() << std::endl
      << "parameters_space: " << std::endl
      << candidate.parameters_space.transpose() << std::endl
      << "reward: " << candidate.reward << std::endl;
}


}
