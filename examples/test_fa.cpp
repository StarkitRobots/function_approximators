#include "rosban_fa/function_approximator_factory.h"
#include "rosban_fa/trainer_factory.h"

#include "rosban_random/tools.h"

using namespace rosban_fa;

Eigen::VectorXd sampleOutput(const Eigen::VectorXd & input,
                             std::default_random_engine * engine) {
  std::normal_distribution<double> noise_distrib(0, 0.001);
  Eigen::VectorXd result(1);
  result(0) = sin(input(0)) + noise_distrib(*engine);
  return result;
}

int main(int argc, char ** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <trainer_config.json>" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::unique_ptr<Trainer> trainer = TrainerFactory().buildFromJsonFile(argv[1]);

  std::default_random_engine engine = rosban_random::getRandomEngine();

  // Getting inputs and observations
  int nb_entries = 500;
  Eigen::MatrixXd limits(1,2);
  limits << -M_PI, M_PI;
  Eigen::MatrixXd inputs, observations;
  inputs = rosban_random::getUniformSamplesMatrix(limits, nb_entries, &engine);
  observations = Eigen::MatrixXd(nb_entries,1);
  for (int i = 0; i < nb_entries; i++) {
    observations.row(i) = sampleOutput(inputs.col(i), &engine);
  }
  // Learning approximator, saving it and reading it to get a copy
  std::unique_ptr<FunctionApproximator> fa = trainer->train(inputs, observations, limits);

  fa->save("fa.bin");

  std::unique_ptr<FunctionApproximator> fa_read;
  FunctionApproximatorFactory().loadFromFile("fa.bin",fa_read);

  // Testing approximators
  int nb_tests = 2;
  Eigen::MatrixXd test_inputs;
  test_inputs = rosban_random::getUniformSamplesMatrix(limits, nb_tests, &engine);
  for (int i = 0; i < test_inputs.cols(); i++) {
    const Eigen::VectorXd & input = test_inputs.col(i);
    Eigen::VectorXd observation;
    Eigen::VectorXd mean_1, mean_2;
    Eigen::MatrixXd covar_1, covar_2;
    observation = sampleOutput(input, &engine);
    fa->predict(input, mean_1, covar_1);
    fa_read->predict(input, mean_2, covar_2);
    // Writing debug
    std::cout << "input: " << input.transpose() << std::endl;
    std::cout << "observation: " << observation.transpose() << std::endl;
    std::cout << "fa         : " << mean_1.transpose() << std::endl;
    std::cout << "fa_read    : " << mean_2.transpose() << std::endl;
  }
}
