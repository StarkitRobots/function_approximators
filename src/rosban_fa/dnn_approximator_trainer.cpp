#include "rosban_fa/dnn_approximator_trainer.h"

#include "rosban_fa/dnn_approximator.h"

#include "rosban_random/tools.h"

using namespace tiny_dnn;

static vec_t cvtEigen2Tiny(const Eigen::VectorXd & v) {
  vec_t tiny(v.rows());
  for (int row = 0; row < v.rows(); row++) {
    tiny[row] = v(row);
  }
  return tiny;
}

static std::vector<vec_t> extractAndCvtEntries(const Eigen::MatrixXd & data,
                                               const std::vector<size_t> & indices) {
  std::vector<vec_t> result(indices.size());
  int dst_idx = 0;
  for (size_t idx : indices) {
    result[dst_idx] = cvtEigen2Tiny(data.col(idx));
    dst_idx++;
  }
  return result;
}

namespace rosban_fa
{

DNNApproximatorTrainer::DNNApproximatorTrainer()
  : layer_units({1}), learning_rate(0.05), nb_minibatches(10), nb_train_epochs(5), cv_ratio(0)
{
}

std::unique_ptr<FunctionApproximator>
DNNApproximatorTrainer::train(const Eigen::MatrixXd & inputs,
                              const Eigen::MatrixXd & observations,
                              const Eigen::MatrixXd & limits) const {
  (void)limits;
  checkConsistency(inputs, observations, limits);
  int input_dim = inputs.rows();
  int output_dim = observations.cols();
  DNNApproximator::network nn = DNNApproximator::buildNN(input_dim, output_dim, layer_units);
  if (verbose) {
    for (size_t i = 0; i < nn.depth(); i++) {
      std::cout << "#layer:" << i << "\n";
      std::cout << "layer type:" << nn[i]->layer_type() << "\n";
      std::cout << "input:" << nn[i]->in_data_size() << "(" << nn[i]->in_data_shape() << ")\n";
      std::cout << "output:" << nn[i]->out_data_size() << "(" << nn[i]->out_data_shape() << ")\n";
    }
  }
  trainNN(&nn, inputs, observations, true);
  // Copy Neural network to get a function approximator
  std::unique_ptr<FunctionApproximator> result(
    new DNNApproximator(nn, input_dim, output_dim, layer_units));
  // Return the function approximator
  return std::move(result);
}

std::unique_ptr<FunctionApproximator>
DNNApproximatorTrainer::train(const Eigen::MatrixXd & inputs,
                              const Eigen::MatrixXd & observations,
                              const Eigen::MatrixXd & limits,
                              const FunctionApproximator & initial_fa) const {
  (void)limits;
  checkConsistency(inputs, observations, limits);
  int input_dim = inputs.rows();
  int output_dim = observations.cols();
  // Initialize network with old approximation
  DNNApproximator::network nn;
  try {
    const DNNApproximator & initial_dnn = dynamic_cast<const DNNApproximator &>(initial_fa);
    nn = initial_dnn.getNetwork();
  } catch (const std::bad_cast & exc) {
    std::cerr << "DNNApproximator::train: expecting DNNApproximator as initial_fa";
    return train(inputs,observations,limits);
  }
  // Train network without weight resets
  trainNN(&nn, inputs, observations, false);
  // Copy Neural network to get a function approximator
  std::unique_ptr<FunctionApproximator> result(
    new DNNApproximator(nn, input_dim, output_dim, layer_units));
  // Return the function approximator
  return std::move(result);
}

void DNNApproximatorTrainer::trainNN(DNNApproximator::network * nn,
                                     const Eigen::MatrixXd & inputs,
                                     const Eigen::MatrixXd & observations,
                                     bool reset_weights) const {
  std::default_random_engine engine = rosban_random::getRandomEngine();
  // Getting dimensions and nb entries
  Eigen::MatrixXd outputs = observations.transpose();
  int nb_entries = inputs.cols();
  // Separating data in training and cross_validation
  size_t nb_entries_cv = std::floor(cv_ratio * nb_entries);
  size_t nb_entries_training = nb_entries - nb_entries_cv;
  std::vector<size_t> set_sizes = {nb_entries_training, nb_entries_cv};
  std::vector<std::vector<size_t>> splitted_indices =
    rosban_random::splitIndices(nb_entries-1, set_sizes, &engine);
  const std::vector<size_t> & training_indices = splitted_indices[0];
  const std::vector<size_t> & cv_indices = splitted_indices[1];
  // Formatting input
  std::vector<vec_t> training_inputs, training_outputs, cv_inputs, cv_outputs;
  training_inputs  = extractAndCvtEntries(inputs , training_indices);
  training_outputs = extractAndCvtEntries(outputs, training_indices);
  cv_inputs  = extractAndCvtEntries(inputs , cv_indices);
  cv_outputs = extractAndCvtEntries(outputs, cv_indices);
  // create callback
  auto on_enumerate_epoch = [&]() {
    if (verbose > 0) {
      double training_loss = nn->get_loss<mse>(training_inputs, training_outputs) / training_inputs.size();
      if (cv_inputs.size() > 0) {
        double cv_loss = nn->get_loss<mse>(cv_inputs, cv_outputs) / cv_inputs.size();
        std::cout << "CV mean loss: " << cv_loss;
      }
      std::cout << " (training mean loss : " << training_loss << ")" << std::endl;
    }
  };
  
  auto on_enumerate_minibatch = [&]() {
  };
  // Launching training
  adam optimizer;
  optimizer.alpha *= learning_rate;
  nn->fit<mse>(optimizer, training_inputs, training_outputs, nb_minibatches, nb_train_epochs,
               on_enumerate_minibatch, on_enumerate_epoch, reset_weights);
}

std::string DNNApproximatorTrainer::getClassName() const {
  return "DNNApproximatorTrainer";
}

Json::Value DNNApproximatorTrainer::toJson() const {
  Json::Value v;
  v["layer_units"] = rhoban_utils::vector2Json(layer_units);
  v["learning_rate"] = learning_rate;
  v["nb_minibatches"] = nb_minibatches;
  v["nb_train_epochs"] = nb_train_epochs;
  v["cv_ratio"] = cv_ratio;
  v["verbose"] = verbose;
  return v;
}

void DNNApproximatorTrainer::fromJson(const Json::Value & v, const std::string & dir_name) {
  (void) dir_name;
  rhoban_utils::tryReadVector(v,"layer_units", &layer_units);
  rhoban_utils::tryRead(v,"learning_rate"  , &learning_rate  );
  rhoban_utils::tryRead(v,"nb_minibatches" , &nb_minibatches );
  rhoban_utils::tryRead(v,"nb_train_epochs", &nb_train_epochs);
  rhoban_utils::tryRead(v,"cv_ratio", &cv_ratio);
  rhoban_utils::tryRead(v,"verbose", &verbose);}
}
