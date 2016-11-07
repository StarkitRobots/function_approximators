#include "rosban_fa/optimizer_trainer_factory.h"

#include "rosban_random/tools.h"

#include <iostream>

using namespace rosban_fa;

struct ParametrizedBlackBox
{
  Eigen::MatrixXd parameters_limits;
  Eigen::MatrixXd actions_limits;
  OptimizerTrainer::RewardFunction reward;
};

/// This function is in a 'n' dimensional space (all dimensions are in -1,1)
/// - parameters: initial position in the 'n' dimensional space
/// - actions: wished movement (same limits as parameters)
/// - noise: real movement is wished_movement * U(1 - noise_ratio, 1 + noise_ratio)
struct ParametrizedBlackBox getContinuousBlackBox(int dim, double noise_ratio)
{
  struct ParametrizedBlackBox result;
  result.parameters_limits = Eigen::MatrixXd(dim,2);
  result.parameters_limits.col(0) = Eigen::VectorXd::Constant(dim, -1);
  result.parameters_limits.col(1) = Eigen::VectorXd::Constant(dim,  1);
  result.actions_limits = result.parameters_limits;
  result.reward = 
    [noise_ratio](const Eigen::VectorXd & parameters,
       const Eigen::VectorXd & actions,
       std::default_random_engine * engine)
    {
      std::uniform_real_distribution<double> noise_distrib(1 - noise_ratio, 1 + noise_ratio);
      double move_ratio = noise_distrib(*engine);
      Eigen::VectorXd final_pos = parameters + actions * move_ratio;
      return -final_pos.squaredNorm();
    };
  return result;
}

int main(int argc, char ** argv)
{
  // Checking parameters
  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " <optimizer_trainer_file>" << std::endl;
    exit(EXIT_FAILURE);
  }
  // Global parameters of execution
  int nb_evaluations = 1000;
  std::default_random_engine engine = rosban_random::getRandomEngine();
  // Read optimizer trainer
  std::unique_ptr<OptimizerTrainer> optimizer_trainer;
  optimizer_trainer = OptimizerTrainerFactory().buildFromXmlFile(argv[1],"OptimizerTrainer");
  // Building pbb and customizing trainer
  struct ParametrizedBlackBox pbb = getContinuousBlackBox(1, 0.05);
  optimizer_trainer->setParametersLimits(pbb.parameters_limits);
  optimizer_trainer->setActionsLimits(pbb.actions_limits);
  // Running optimizer trainer
  std::unique_ptr<FunctionApproximator> policy;
  policy = optimizer_trainer->train(pbb.reward, &engine);
  // Saving policy
  policy->save("policy.bin");
  // Performing evaluation
  double total_reward = 0;
  Eigen::MatrixXd evaluation_set;
  evaluation_set = rosban_random::getUniformSamplesMatrix(pbb.parameters_limits,
                                                          nb_evaluations,
                                                          &engine);
  for (int i = 0; i < nb_evaluations; i++)
  {
    Eigen::VectorXd parameters = evaluation_set.col(i);
    Eigen::VectorXd action;
    Eigen::MatrixXd covar;
    policy->predict(parameters, action, covar);
    total_reward += pbb.reward(parameters, action, &engine);
  }
  double avg_reward = total_reward / nb_evaluations;
  std::cout << "Average reward: " << avg_reward;
}
