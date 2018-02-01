#include "rosban_fa/optimizer_trainer_factory.h"

#include "rhoban_random/tools.h"

#include <iostream>

#include <fenv.h>


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
struct ParametrizedBlackBox getContinuousBlackBox(int dim, double noise_ratio, double tol)
{
  struct ParametrizedBlackBox result;
  result.parameters_limits = Eigen::MatrixXd(dim,2);
  result.parameters_limits.col(0) = Eigen::VectorXd::Constant(dim, -1);
  result.parameters_limits.col(1) = Eigen::VectorXd::Constant(dim,  1);
  result.actions_limits = result.parameters_limits;
  result.reward = 
    [noise_ratio, tol](const Eigen::VectorXd & parameters,
       const Eigen::VectorXd & actions,
       std::default_random_engine * engine)
    {
      std::uniform_real_distribution<double> noise_distrib(1 - noise_ratio, 1 + noise_ratio);
      double move_ratio = noise_distrib(*engine);
      Eigen::VectorXd final_pos = parameters + actions * move_ratio;
      double error = std::sqrt(final_pos.squaredNorm());
      double extra_error = std::max(0.0,error - tol);
      return -(extra_error * extra_error);
    };
  return result;
}

// NOT WORKING YET
//struct ParametrizedBlackBox getDiscreteBlackBox(int dim)
//{
//  struct ParametrizedBlackBox result;
//  result.parameters_limits = Eigen::MatrixXd(dim,2);
//  result.parameters_limits.col(0) = Eigen::VectorXd::Constant(dim, -1);
//  result.parameters_limits.col(1) = Eigen::VectorXd::Constant(dim,  1);
//  result.actions_limits = result.parameters_limits;
//  result.reward = 
//    [noise_ratio, tol](const Eigen::VectorXd & parameters,
//       const Eigen::VectorXd & actions,
//       std::default_random_engine * engine)
//    {
//      std::uniform_real_distribution<double> noise_distrib(1 - noise_ratio, 1 + noise_ratio);
//      double move_ratio = noise_distrib(*engine);
//      Eigen::VectorXd final_pos = parameters + actions * move_ratio;
//      double error = std::sqrt(final_pos.squaredNorm());
//      double extra_error = std::max(0.0,error - tol);
//      return -(extra_error * extra_error);
//    };
//  return result;
//}

int main(int argc, char ** argv)
{
  feenableexcept(FE_DIVBYZERO| FE_INVALID | FE_OVERFLOW);

  // Checking parameters
  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " <optimizer_trainer_file>" << std::endl;
    exit(EXIT_FAILURE);
  }
  // Global parameters of execution
  int nb_evaluations = 1000;
  std::default_random_engine engine = rhoban_random::getRandomEngine();
  // Read optimizer trainer
  std::unique_ptr<OptimizerTrainer> optimizer_trainer;
  optimizer_trainer = OptimizerTrainerFactory().buildFromJsonFile(argv[1]);
  // Building pbb and customizing trainer
  struct ParametrizedBlackBox pbb = getContinuousBlackBox(2, 0.05,0.1);
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
  evaluation_set = rhoban_random::getUniformSamplesMatrix(pbb.parameters_limits,
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
  std::cout << "Average reward: " << avg_reward << std::endl;
}
