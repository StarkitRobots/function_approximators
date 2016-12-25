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
#include "rosban_utils/time_stamp.h"

using rosban_utils::TimeStamp;

namespace rosban_fa
{

AdaptativeTree::AdaptativeTree()
  : nb_generations(1),
    nb_samples(100),
    cv_ratio(1.0),
    evaluation_trials(10),
    nb_samples_treated(0),
    verbosity(2),
    reuse_samples(true),
    use_point_splits(false),
    max_depth(-1),
    narrow_linear_slope(false),
    constant_uses_guess(false),
    linear_uses_guess(false)
{
}

AdaptativeTree::~AdaptativeTree() {}

std::unique_ptr<FunctionApproximator>
AdaptativeTree::train(RewardFunction rf, std::default_random_engine * engine)
{
  if (max_depth < 0 && !reuse_samples) {
    std::cout << "WARNING: no max_depth and not reusing samples!!!!"
              << " This might lead to infinite loop" << std::endl;
  }
  std::unique_ptr<FunctionApproximator> result;
  for (int generation = 1; generation <= nb_generations; generation++)
  {
    if (verbosity >= 1) {
      std::cout << "Generation " << generation << "/" << nb_generations
                << std::endl;
    }
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
      // Affecting samples
      parameters_set.block(0, sample_id,
                           param_dims, leaf_nb_samples) = leaf_samples;
      sample_id += leaf_nb_samples;
    }
  }
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
  candidate.depth = 0;
  Eigen::VectorXd guess;// No specific guess on first node
  updateAction(rf, candidate, guess, engine);
  updateReward(rf, candidate, engine);
  return buildApproximator(rf, candidate, engine);
}

std::unique_ptr<FunctionApproximator>
AdaptativeTree::buildApproximator(RewardFunction rf,
                                  ApproximatorCandidate & candidate,
                                  std::default_random_engine * engine)
{
  TimeStamp build_approximator_start = TimeStamp::now();

  // Initializing variables
  double best_reward = candidate.reward;
  std::unique_ptr<Split> best_split;
  std::vector<ApproximatorCandidate> childs;
  // Getting splits
  TimeStamp split_candidate_start = TimeStamp::now();
  std::vector<std::unique_ptr<Split>> split_candidates;

  // Disabling candidates if above max_depth
  if (max_depth < 0 || max_depth > candidate.depth) { 
    split_candidates = getSplitCandidates(candidate.parameters_set);
    TimeStamp split_candidate_end = TimeStamp::now();
    std::cout << "Time spent to get split candidates: "
              << diffSec(split_candidate_start, split_candidate_end) << " s"
              << std::endl;
  }

  int candidate_samples = candidate.parameters_set.cols();

  if (verbosity >= 3) {
    std::cout << "buildApproximator:" << std::endl
              << "space:" << std::endl
              << candidate.parameters_space.transpose() << std::endl
              << "candidate_samples: " << candidate_samples << std::endl;
  }

  // Checking if a split improves reward criteria
  for (size_t split_idx = 0; split_idx < split_candidates.size(); split_idx++)
  {
    std::unique_ptr<Split> current_split = std::move(split_candidates[split_idx]);
    if (verbosity >= 3) {
      std::cout << "\tEvaluating split: " << current_split->toString() << std::endl;
    }
    // Stored data for evaluation of the split
    std::vector<Eigen::MatrixXd> spaces, samples;
    std::vector<double> rewards;
    std::vector<std::unique_ptr<FunctionApproximator>> function_approximators;
    // Separate elements and space with reward
    int nb_elements = current_split->getNbElements();
    spaces = current_split->splitSpace(candidate.parameters_space);
    // If samples are generated once for every generation, distribute them
    if (reuse_samples) {
      samples = current_split->splitEntries(candidate.parameters_set);
      // If a split would result on getting one of the space empty, refuse it
      bool creates_empty_spaces = false;
      for (const Eigen::MatrixXd & samples_set : samples) {
        if (samples_set.cols() == 0) {
          creates_empty_spaces = true;
          break;
        }
      }
      if (creates_empty_spaces) {
        if (verbosity >= 3) {
          std::cout << "\t-> Creating empty spaces: refused" << std::endl;
        }
        continue;
      }
    }
    else {
      for (size_t elem_id = 0; elem_id < spaces.size(); elem_id++) {
        samples.push_back(rosban_random::getUniformSamplesMatrix(spaces[elem_id],
                                                                 nb_samples,
                                                                 engine));
      }
    }
    // Estimate rewards and function approximators for all elements of the split
    double total_reward = 0;
    double total_weight = 0;
    for (int elem_id = 0; elem_id < nb_elements; elem_id++)
    {
      // Initializing candidate
      ApproximatorCandidate current_candidate;
      current_candidate.parameters_set = samples[elem_id];
      current_candidate.parameters_space = spaces[elem_id];
      // Guess the action using the splitted node evaluation
      Eigen::VectorXd space_center = (spaces[elem_id].col(0) + spaces[elem_id].col(1)) / 2;
      Eigen::VectorXd guessed_action = candidate.approximator->predict(space_center);
      if (verbosity >= 3) {
        std::cout << "\t# Elem " << (elem_id+1) << "/" << nb_elements << std::endl;
        std::cout << "\t\tSpace:" << std::endl
                  << current_candidate.parameters_space.transpose() << std::endl;
      }
      // Optimizing action
      updateAction(rf, current_candidate, guessed_action, engine);
      updateReward(rf, current_candidate, engine);
      // Storing values
      function_approximators.push_back(std::move(current_candidate.approximator));
      rewards.push_back(current_candidate.reward);
      // Updating values
      // TODO: validate weighting method
      double weight = samples[elem_id].cols();
      total_reward += current_candidate.reward * weight;
      total_weight += weight;
      if (verbosity >= 3) {
        std::cout << "\t\tAvg Reward: " << current_candidate.reward << std::endl;
        std::cout << "\t\tWeight: " << weight << std::endl;
      }
    }
    double avg_split_reward = total_reward / total_weight;
    // If split is currently the best, store its internal data
    if (avg_split_reward > best_reward)
    {
      if (verbosity >= 3) {
        std::cout << "\t\t<- Best split found yet" << std::endl;
      }
      best_reward = avg_split_reward;
      childs.clear();
      childs.resize(nb_elements);
      for (int elem_id = 0; elem_id < nb_elements; elem_id++)
      {
        childs[elem_id].approximator = std::move(function_approximators[elem_id]);
        childs[elem_id].parameters_set = samples[elem_id];
        childs[elem_id].parameters_space = spaces[elem_id];
        childs[elem_id].reward = rewards[elem_id];
        childs[elem_id].depth = candidate.depth + 1;
      }
      best_split = std::move(current_split);
    }
  }
  // If no interesting split has been fonund
  if (childs.size() == 0) {
    // Debug print
    double elapsed = diffSec(build_approximator_start, TimeStamp::now());
    std::cout << "buildApproximator:"
              << nb_samples << ","
              << elapsed << std::endl;
    if (verbosity >= 2) {
      std::cout << "Leaf reached at depth " << candidate.depth << std::endl;
      nb_samples_treated += candidate.parameters_set.cols();
      print(candidate, std::cout);
    }
    if (verbosity >= 1 && reuse_samples) {
      std::cout << "Samples treated: " << nb_samples_treated << "/" 
                << nb_samples << std::endl;
    }
    // filling processed_leaves
    ProcessedLeaf leaf;
    double custom_reward = 0;//TODO
    leaf.space = candidate.parameters_space;
    leaf.loss = candidate.reward - custom_reward;
    leaf.nb_samples = candidate.parameters_set.size();
    processed_leaves.push_back(leaf);
    return std::move(candidate.approximator);
  }
  // If an interesting split has been found, then become a node with
  // function_approximators as childs
  else {
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
  if (use_point_splits && samples.cols() >= std::pow(2, getParametersDim())) {
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

void AdaptativeTree::updateAction(RewardFunction rf,
                                  ApproximatorCandidate & candidate,
                                  const Eigen::VectorXd & guess,
                                  std::default_random_engine * engine)
{
  const Eigen::MatrixXd & training_set = candidate.parameters_set;
  const Eigen::MatrixXd & parameters_space = candidate.parameters_space;
  Eigen::VectorXd space_center = (parameters_space.col(0) + parameters_space.col(1)) / 2;
  EvaluationFunction policy_evaluator = getEvaluationFunction(rf, training_set);
  std::unique_ptr<FunctionApproximator> constant_policy;
  // Default is constant policy:
  TimeStamp constant_start = TimeStamp::now();
  constant_policy = optimizeConstantPolicy(policy_evaluator, guess, engine);
  double constant_score = policy_evaluator(*constant_policy, engine);
  TimeStamp constant_end = TimeStamp::now();
  std::cout << "updateAction:ConstantTrain,"
            << training_set.cols() << ","
            << diffSec(constant_start, constant_end) << std::endl;
  // Train linear model if there is no risk of underconstrained learning
  // - number of observations is: |A| * nb_samples
  // - number of parameters is  : |A| * (param_dims + 1)
  if (training_set.cols() >= (getParametersDim() + 1)) {
    std::unique_ptr<FunctionApproximator> linear_policy;
    linear_policy = optimizeLinearPolicy(policy_evaluator,
                                         parameters_space,
                                         constant_policy->predict(space_center),
                                         engine);
    double linear_score = policy_evaluator(*linear_policy, engine);
    TimeStamp linear_end = TimeStamp::now();
    std::cout << "updateAction:LinearTrain,"
              << training_set.cols() << ","
              << diffSec(constant_end, linear_end) << std::endl;
    // Returning linear if it is better
    if (linear_score > constant_score) {
      candidate.approximator = std::move(linear_policy);
      return;
    }
  }
  // If we failed to train a linear model or if constant model was more suited
  candidate.approximator = std::move(constant_policy);
}

std::unique_ptr<FunctionApproximator>
AdaptativeTree::optimizeConstantPolicy(EvaluationFunction policy_evaluator,
                                       const Eigen::VectorXd & initial_guess,
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

  Eigen::VectorXd init_params = initial_guess;
  // If guess is forbidden or has not been provided, use the center of the action space
  if (!constant_uses_guess || init_params.rows() < 1) {
    init_params = (actions_limits.col(0) + actions_limits.col(1)) / 2;
  }

  if (verbosity >= 3) {
    std::cout << "training a constant model with initial guess: "
              << init_params.transpose() << std::endl;
  }

  // Training a constant model
  model_optimizer->setLimits(actions_limits);
  Eigen::VectorXd best_action = model_optimizer->train(constant_model_reward_func,
                                                       init_params, engine);
  return std::unique_ptr<FunctionApproximator>(new ConstantApproximator(best_action));
}

std::unique_ptr<FunctionApproximator>
AdaptativeTree::optimizeLinearPolicy(EvaluationFunction policy_evaluator,
                                     const Eigen::MatrixXd & parameters_space,
                                     const Eigen::VectorXd & guess,
                                     std::default_random_engine * engine)
{
  if (!model_optimizer) {
    throw std::runtime_error("AdaptativeTree::optimizeLinearPolicy: No model optimizer available");
  }

  int parameter_dims = getParametersDim();
  int action_dims = getActionsDim();

  int training_dims = (parameter_dims +1) * action_dims;
  // Initial parameters
  Eigen::VectorXd initial_params;
  initial_params = LinearApproximator::getDefaultParameters(parameters_limits,
                                                            actions_limits);
  // Uses of guess can be enabled or disabled
  if (linear_uses_guess) {
    // Set the bias according to the guess
    initial_params.segment(0,action_dims) = guess;
  }

  // Creating linear parameters space
  Eigen::MatrixXd linear_parameters_space;
  linear_parameters_space = LinearApproximator::getParametersSpace(parameters_limits,
                                                                   actions_limits,
                                                                   narrow_linear_slope);
  // Getting center
  Eigen::VectorXd params_center;
  params_center = (parameters_space.col(1) + parameters_space.col(0)) / 2;

  // Creating the reward function for linear models
  rosban_bbo::Optimizer::RewardFunc linear_model_reward_func;
  linear_model_reward_func =
    [policy_evaluator, action_dims, parameter_dims, params_center]
    (const Eigen::VectorXd & parameters, std::default_random_engine * engine)
    {
      LinearApproximator policy(parameter_dims, action_dims, parameters, params_center);
      return policy_evaluator(policy, engine);
    };

  if (verbosity >= 3) {
    std::cout << "training a linear model with initial guess: "
              << initial_params.transpose() << std::endl
              << "Linear parameters space: " << std::endl
              << linear_parameters_space.transpose() << std::endl;
  }

  // Training a linear model
  model_optimizer->setLimits(linear_parameters_space);
  Eigen::VectorXd best_action = model_optimizer->train(linear_model_reward_func,
                                                       initial_params,
                                                       engine);
  return std::unique_ptr<FunctionApproximator>
    (new LinearApproximator(parameter_dims, action_dims, best_action, params_center));
}


std::string AdaptativeTree::class_name() const
{
  return "AdaptativeTree";
}

void AdaptativeTree::to_xml(std::ostream &out) const
{
  rosban_utils::xml_tools::write<int>   ("nb_generations"     , nb_generations     , out);
  rosban_utils::xml_tools::write<int>   ("nb_samples"         , nb_samples         , out);
  rosban_utils::xml_tools::write<int>   ("evaluation_trials"  , evaluation_trials  , out);
  rosban_utils::xml_tools::write<int>   ("verbosity"          , verbosity          , out);
  rosban_utils::xml_tools::write<int>   ("max_depth"          , max_depth          , out);
  rosban_utils::xml_tools::write<bool>  ("reuse_samples"      , reuse_samples      , out);
  rosban_utils::xml_tools::write<bool>  ("use_point_splits"   , use_point_splits   , out);
  rosban_utils::xml_tools::write<bool>  ("narrow_linear_slope", narrow_linear_slope, out);
  rosban_utils::xml_tools::write<bool>  ("constant_uses_guess", constant_uses_guess, out);
  rosban_utils::xml_tools::write<bool>  ("linear_uses_guess"  , linear_uses_guess  , out);
  rosban_utils::xml_tools::write<double>("cv_ratio"           , cv_ratio           , out);
}

void AdaptativeTree::from_xml(TiXmlNode *node)
{
  rosban_utils::xml_tools::try_read<int>   (node, "nb_generations"     , nb_generations     );
  rosban_utils::xml_tools::try_read<int>   (node, "nb_samples"         , nb_samples         );
  rosban_utils::xml_tools::try_read<int>   (node, "evaluation_trials"  , evaluation_trials  );
  rosban_utils::xml_tools::try_read<int>   (node, "verbosity"          , verbosity          );
  rosban_utils::xml_tools::try_read<int>   (node, "max_depth"          , max_depth          );
  rosban_utils::xml_tools::try_read<bool>  (node, "reuse_samples"      , reuse_samples      );
  rosban_utils::xml_tools::try_read<bool>  (node, "use_point_splits"   , use_point_splits   );
  rosban_utils::xml_tools::try_read<bool>  (node, "narrow_linear_slope", narrow_linear_slope);
  rosban_utils::xml_tools::try_read<bool>  (node, "constant_uses_guess", constant_uses_guess);
  rosban_utils::xml_tools::try_read<bool>  (node, "linear_uses_guess"  , linear_uses_guess  );
  rosban_utils::xml_tools::try_read<double>(node, "cv_ratio"           , cv_ratio           );
  rosban_bbo::OptimizerFactory().tryRead   (node,"model_optimizer", model_optimizer);
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
