#include "starkit_fa/function_approximator.h"
#include "starkit_fa/function_approximator_factory.h"
#include "starkit_fa/trainer.h"
#include "starkit_fa/trainer_factory.h"

#include <fstream>
#include <iostream>

using starkit_fa::FunctionApproximator;
using starkit_fa::FunctionApproximatorFactory;
using starkit_fa::Trainer;
using starkit_fa::TrainerFactory;

int main()
{
  // Creating some random samples for test
  int nb_cells = 5;
  int nb_cells2 = nb_cells * nb_cells;
  Eigen::MatrixXd inputs(2, nb_cells2);
  Eigen::MatrixXd observations(nb_cells2, 2);
  Eigen::MatrixXd limits(2, 2);
  limits << 0, nb_cells, 0, nb_cells;
  // Creating sample from (x,y) -> (x - 2 * y, abs(x) * sin(y))
  int sample = 0;
  for (int x = 0; x < nb_cells; x++)
  {
    for (int y = 0; y < nb_cells; y++)
    {
      inputs(0, sample) = x;
      inputs(1, sample) = y;
      observations(sample, 0) = x - 2 * y;
      observations(sample, 1) = std::fabs(x) - sin(y);
      sample++;
    }
  }
  std::vector<std::string> test_types = { "GPTrainer", "GPForestTrainer", "PWCForestTrainer", "PWLForestTrainer" };
  TrainerFactory trainer_factory;
  FunctionApproximatorFactory fa_factory;
  for (const std::string& trainer_name : test_types)
  {
    std::cout << "-------------------------------------------" << std::endl
              << "Running test for trainer: " << trainer_name << std::endl;

    // Create a trainer
    std::unique_ptr<Trainer> trainer = trainer_factory.build(trainer_name);

    std::unique_ptr<FunctionApproximator> original_fa, copy_fa;
    original_fa = trainer->train(inputs, observations, limits);

    // writing original
    std::string filename("/tmp/fa.bin");
    std::string copy_filename("/tmp/copy_fa.bin");
    int original_bytes_written = original_fa->save(filename);
    std::cout << "Original bytes written: " << original_bytes_written << std::endl;

    // Reading from original
    int original_bytes_read = fa_factory.loadFromFile(filename, copy_fa);
    std::cout << "Original bytes read   : ";
    std::cout << original_bytes_read;
    std::cout << std::endl;

    // writing copy
    int copy_bytes_written = copy_fa->save(copy_filename);
    std::cout << "Copy bytes written    : " << copy_bytes_written << std::endl;

    Eigen::VectorXd test_input(2);
    test_input << 1.5, 2.5;

    Eigen::VectorXd original_prediction, copy_prediction;
    Eigen::MatrixXd original_variance, copy_variance;
    original_fa->predict(test_input, original_prediction, original_variance);
    copy_fa->predict(test_input, copy_prediction, copy_variance);

    // Outputting some messages:
    std::cout << "For test input: " << test_input.transpose() << std::endl
              << "\toriginal prediction : " << original_prediction.transpose() << std::endl
              << "\tcopy prediction     : " << copy_prediction.transpose() << std::endl
              << "\toriginal variance   : " << std::endl
              << original_variance << std::endl
              << "\tcopy variance       : " << std::endl
              << copy_variance << std::endl;
  }
}
