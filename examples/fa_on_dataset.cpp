#include "starkit_fa/function_approximator_factory.h"
#include "starkit_fa/trainer_factory.h"

#include "starkit_random/tools.h"

using namespace starkit_fa;
using namespace starkit_utils;

class DataSet : public JsonSerializable
{
public:
  DataSet()
  {
  }

  std::string getClassName() const override
  {
    return "DataSet";
  }

  void fromJson(const Json::Value& v, const std::string& dir_name) override
  {
    (void)dir_name;
    inputs = starkit_utils::read<Eigen::MatrixXd>(v, "inputs");
    outputs = starkit_utils::read<Eigen::MatrixXd>(v, "outputs");
  }
  Json::Value toJson() const override
  {
    Json::Value v;
    v["inputs"] = matrix2Json(inputs);
    v["outputs"] = matrix2Json(outputs);
    return v;
  }

  Eigen::MatrixXd inputs;
  Eigen::MatrixXd outputs;
};

int main(int argc, char** argv)
{
  if (argc < 3)
  {
    std::cerr << "Usage: " << argv[0] << " <trainer_config.json> <dataset.json>" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string trainer_path(argv[1]);
  std::string dataset_path(argv[2]);

  std::unique_ptr<Trainer> trainer = TrainerFactory().buildFromJsonFile(trainer_path);
  DataSet ds;
  ds.loadFile(dataset_path, "./");

  // Learning approximator, saving it and reading it to get a copy
  // TODO: limits should be provided by the dataset
  std::unique_ptr<FunctionApproximator> fa = trainer->train(ds.inputs, ds.outputs.transpose(), Eigen::MatrixXd(10, 2));

  fa->save("fa.bin");

  std::unique_ptr<FunctionApproximator> fa_read;
  FunctionApproximatorFactory().loadFromFile("fa.bin", fa_read);

  // Testing approximators on inputs
  int nb_tests = 5;
  for (int i = 0; i < nb_tests; i++)
  {
    const Eigen::VectorXd& input = ds.inputs.col(i);
    Eigen::VectorXd observation;
    Eigen::VectorXd mean_1, mean_2;
    Eigen::MatrixXd covar_1, covar_2;
    observation = ds.outputs.col(i);
    fa->predict(input, mean_1, covar_1);
    fa_read->predict(input, mean_2, covar_2);
    // Writing debug
    std::cout << "--------------------" << std::endl;
    std::cout << "input: " << input.transpose() << std::endl;
    std::cout << "observation: " << observation.transpose() << std::endl;
    std::cout << "fa         : " << mean_1.transpose() << std::endl;
    std::cout << "fa_read    : " << mean_2.transpose() << std::endl;
  }
}
