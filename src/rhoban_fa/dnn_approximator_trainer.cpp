#include "rhoban_fa/dnn_approximator_trainer.h"

#include "rhoban_fa/dnn_approximator.h"

#include "rhoban_random/tools.h"

#include "rhoban_utils/threading/multi_core.h"

using namespace tiny_dnn;

namespace rhoban_fa
{

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

static std::string loss2Str(DNNApproximatorTrainer::LossFunction loss) {
  switch(loss) {
    case DNNApproximatorTrainer::LossFunction::MSE: return "MSE";
    case DNNApproximatorTrainer::LossFunction::Abs: return "Abs";
  }
  throw std::logic_error("loss2Str: unknown value for loss");
}

static DNNApproximatorTrainer::LossFunction str2Loss(const std::string & str) {
  if (str == "MSE") return DNNApproximatorTrainer::LossFunction::MSE;
  if (str == "Abs") return DNNApproximatorTrainer::LossFunction::Abs;
  throw std::logic_error("str2Loss: unknown value '" + str + "'");
}

DNNApproximatorTrainer::DNNApproximatorTrainer()
  : learning_rates({1}), nb_minibatches(10), nb_train_epochs(5), cv_ratio(0),
    loss(LossFunction::MSE), verbose(0)
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
  DNNApproximator::network nn = DNNApproximator::buildNN(input_dim, output_dim, dnn_config);
  if (verbose) {
    std::cout << "DNN Structure: " << std::endl;
    for (size_t i = 0; i < nn.depth(); i++) {
      std::cout << "#layer:" << i << "\n";
      std::cout << "layer type:" << nn[i]->layer_type() << "\n";
      std::cout << "input:" << nn[i]->in_data_size() << "(" << nn[i]->in_data_shape() << ")\n";
      std::cout << "output:" << nn[i]->out_data_size() << "(" << nn[i]->out_data_shape() << ")\n";
    }
  }
  nn = trainBestNN(nn, inputs, observations.transpose(), true);
  // Copy Neural network to get a function approximator
  std::unique_ptr<FunctionApproximator> result(
    new DNNApproximator(nn, input_dim, output_dim, dnn_config));
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
  nn = trainBestNN(nn, inputs, observations.transpose(), false);
  // Copy Neural network to get a function approximator
  std::unique_ptr<FunctionApproximator> result(
    new DNNApproximator(nn, input_dim, output_dim, dnn_config));
  // Return the function approximator
  return std::move(result);
}

DNNApproximator::network
DNNApproximatorTrainer::trainBestNN(const DNNApproximator::network & initial_network,
                                    const Eigen::MatrixXd & inputs,
                                    const Eigen::MatrixXd & outputs,
                                    bool reset_weights) const
{
  // Creating inputs and outputs
  std::default_random_engine engine = rhoban_random::getRandomEngine();
  // Getting dimensions and nb entries
  int nb_entries = inputs.cols();
  // Separating data in training and cross_validation
  size_t nb_entries_cv = std::floor(cv_ratio * nb_entries);
  size_t nb_entries_training = nb_entries - nb_entries_cv;
  std::vector<size_t> set_sizes = {nb_entries_training, nb_entries_cv};
  std::vector<std::vector<size_t>> splitted_indices =
    rhoban_random::splitIndices(nb_entries-1, set_sizes, &engine);
  const std::vector<size_t> & training_indices = splitted_indices[0];
  const std::vector<size_t> & cv_indices = splitted_indices[1];
  // Formatting input
  std::vector<vec_t> training_inputs, training_outputs, cv_inputs, cv_outputs;
  training_inputs  = extractAndCvtEntries(inputs , training_indices);
  training_outputs = extractAndCvtEntries(outputs, training_indices);
  cv_inputs  = extractAndCvtEntries(inputs , cv_indices);
  cv_outputs = extractAndCvtEntries(outputs, cv_indices);
  // Preparing multiple copies of networks and cv_values to learn simultaneously
  // TinyDNN copy of network is not a 'real' in depth copy, therefore, multi-threading
  // issues occurs when using the copy constructor
  //
  // TODO: either reimplement network copy, or at least use a randomly generated
  // name and check if it exists in order to avoid collision of files
  initial_network.save(".tmp_file.data");
  std::vector<DNNApproximator::network> networks(learning_rates.size());
  for (size_t idx = 0; idx < learning_rates.size(); idx++) {
    networks[idx].load(".tmp_file.data");
  }
  std::vector<double> cv_losses(learning_rates.size());
  auto learning_task =
    [&](int start_idx, int end_idx)
    {
      for (int idx = start_idx; idx < end_idx; idx++) {
        std::cout << "Training " << idx << std::endl;
        trainNN(training_inputs, training_outputs, cv_inputs, cv_outputs,
                learning_rates[idx], reset_weights, &(networks[idx]), &(cv_losses[idx]));
      }
    };
  rhoban_utils::MultiCore::runParallelTask(learning_task, learning_rates.size(), nb_threads);
  // Select best neural network based on value
  double best_cv = std::numeric_limits<double>::max();
  int best_cv_idx = -1;
  for (size_t idx = 0; idx < learning_rates.size(); idx++) {
    if (verbose > 0) {
      std::cout << "CV[" << idx << "] : " << cv_losses[idx]
                << "(learning rate: " << learning_rates[idx] << ")" << std::endl;
    }
    if (cv_losses[idx] < best_cv) {
      best_cv_idx = idx;
      best_cv = cv_losses[idx];
    }
  }
  return networks[best_cv_idx];
}

double DNNApproximatorTrainer::getLoss(DNNApproximator::network * nn,
                                       const std::vector<tiny_dnn::vec_t> & inputs,
                                       const std::vector<tiny_dnn::vec_t> & outputs) const {
  switch(loss) {
    case LossFunction::MSE:
      return nn->get_loss<mse>(inputs, outputs) / inputs.size();
    case LossFunction::Abs:
      return nn->get_loss<absolute>(inputs, outputs) / inputs.size();
  }
  throw std::logic_error("DNNApproximatorTrainer::getLoss: unknown loss function");
}

void DNNApproximatorTrainer::trainNN(const std::vector<vec_t> & training_inputs,
                                     const std::vector<vec_t> & training_outputs,
                                     const std::vector<vec_t> & cv_inputs,
                                     const std::vector<vec_t> & cv_outputs,
                                     double learning_rate,
                                     bool reset_weights,
                                     DNNApproximator::network * nn,
                                     double * cv_loss) const {
  // create callback
  auto on_enumerate_epoch = [&]() {
    *cv_loss = getLoss(nn, cv_inputs, cv_outputs);
    if (verbose > 0) {
      double training_loss = getLoss(nn, training_inputs, training_outputs);
      std::cout << "[LR: " << learning_rate << "] CV mean loss: " << (*cv_loss)
                << " (training mean loss : " << training_loss << ")" << std::endl;
    }
  };
  
  auto on_enumerate_minibatch = [&]() {
  };

  // Launching training
  adam optimizer;
  optimizer.alpha *= learning_rate;
  switch (loss) {
    case LossFunction::MSE:
      nn->fit<mse>(optimizer, training_inputs, training_outputs,
                   nb_minibatches, nb_train_epochs,
                   on_enumerate_minibatch, on_enumerate_epoch, reset_weights, 1);
      break;
    case LossFunction::Abs:
      nn->fit<absolute>(optimizer, training_inputs, training_outputs,
                        nb_minibatches, nb_train_epochs,
                        on_enumerate_minibatch, on_enumerate_epoch, reset_weights, 1);
      break;
  }
}

std::string DNNApproximatorTrainer::getClassName() const {
  return "DNNApproximatorTrainer";
}

Json::Value DNNApproximatorTrainer::toJson() const {
  Json::Value v = Trainer::toJson();
  v["dnn_config"] = dnn_config.toJson();
  v["learning_rates"] = rhoban_utils::vector2Json(learning_rates);
  v["nb_minibatches"] = nb_minibatches;
  v["nb_train_epochs"] = nb_train_epochs;
  v["cv_ratio"] = cv_ratio;
  v["loss"] = loss2Str(loss);
  v["verbose"] = verbose;
  return v;
}

void DNNApproximatorTrainer::fromJson(const Json::Value & v, const std::string & dir_name) {
  Trainer::fromJson(v, dir_name);
  std::string loss_str;
  dnn_config.tryRead(v, "dnn_config", dir_name);
  rhoban_utils::tryReadVector(v,"learning_rates", &learning_rates);
  rhoban_utils::tryRead(v,"nb_minibatches" , &nb_minibatches );
  rhoban_utils::tryRead(v,"nb_train_epochs", &nb_train_epochs);
  rhoban_utils::tryRead(v,"cv_ratio", &cv_ratio);
  rhoban_utils::tryRead(v,"loss", &loss_str);
  rhoban_utils::tryRead(v,"verbose", &verbose);
  if (loss_str!="") loss = str2Loss(loss_str);
}

}
