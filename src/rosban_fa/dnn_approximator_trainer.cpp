#include "rosban_fa/dnn_approximator_trainer.h"

#include "rosban_fa/dnn_approximator.h"

using namespace tiny_dnn;

static vec_t cvtEigen2Tiny(const Eigen::VectorXd & v) {
  vec_t tiny(v.rows());
  for (int row = 0; row < v.rows(); row++) {
    tiny[row] = v(row);
  }
  return tiny;
}

namespace rosban_fa
{

DNNApproximatorTrainer::DNNApproximatorTrainer()
  : nb_units(1), learning_rate(0.05), nb_minibatches(10), nb_train_epochs(5)
{
}

std::unique_ptr<FunctionApproximator>
DNNApproximatorTrainer::train(const Eigen::MatrixXd & inputs,
                              const Eigen::MatrixXd & observations,
                              const Eigen::MatrixXd & limits) const {
  checkConsistency(inputs, observations, limits);
  // Getting dimensions and nb entries
  int nb_entries = inputs.cols();
  int input_dim = inputs.rows();
  int output_dim = observations.cols();
  // Formatting input
  std::vector<vec_t> formatted_inputs(nb_entries), formatted_outputs(nb_entries);
  for (int idx = 0; idx < nb_entries; idx++) {
    formatted_inputs[idx] = cvtEigen2Tiny(inputs.col(idx));
    formatted_outputs[idx] = cvtEigen2Tiny(observations.row(idx));
  }
  // Running network optimization
  adam optimizer;
  optimizer.alpha *= learning_rate;
  DNNApproximator::network nn = DNNApproximator::buildNN(input_dim, output_dim, nb_units);
  for (size_t i = 0; i < nn.depth(); i++) {
    std::cout << "#layer:" << i << "\n";
    std::cout << "layer type:" << nn[i]->layer_type() << "\n";
    std::cout << "input:" << nn[i]->in_data_size() << "(" << nn[i]->in_data_shape() << ")\n";
    std::cout << "output:" << nn[i]->out_data_size() << "(" << nn[i]->out_data_shape() << ")\n";
  }
  // create callback
  auto on_enumerate_epoch = [&]() {
    double loss = nn.get_loss<mse>(formatted_inputs, formatted_outputs);
    std::cout << "Loss: " << loss << std::endl;
  };
  
  auto on_enumerate_minibatch = [&]() {
  };
  // Launching training
  nn.fit<mse>(optimizer, formatted_inputs, formatted_outputs, nb_minibatches, nb_train_epochs,
              on_enumerate_minibatch, on_enumerate_epoch);
  DNNApproximator::network nncp = nn;
  // Copy Neural network to get a function approximator
  std::unique_ptr<FunctionApproximator> result(new DNNApproximator(nn, input_dim,
                                                                   output_dim, nb_units));
  // Return the function approximator
  return std::move(result);
}

std::string DNNApproximatorTrainer::getClassName() const {
  return "DNNApproximatorTrainer";
}

Json::Value DNNApproximatorTrainer::toJson() const {
  Json::Value v;
  v["nb_units"] = nb_units;
  v["learning_rate"] = learning_rate;
  v["nb_minibatches"] = nb_minibatches;
  v["nb_train_epochs"] = nb_train_epochs;
  return v;
}

void DNNApproximatorTrainer::fromJson(const Json::Value & v, const std::string & dir_name) {
  (void) dir_name;
  rhoban_utils::tryRead(v,"nb_units"       , &nb_units       );
  rhoban_utils::tryRead(v,"learning_rate"  , &learning_rate  );
  rhoban_utils::tryRead(v,"nb_minibatches" , &nb_minibatches );
  rhoban_utils::tryRead(v,"nb_train_epochs", &nb_train_epochs);
}

}
