#pragma once

#include "rosban_fa/trainer.h"

#include "rosban_fa/dnn_approximator.h"

namespace rosban_fa
{

class DNNApproximatorTrainer : public Trainer {
public:

  enum LossFunction {MSE, Abs};

  DNNApproximatorTrainer();

  virtual std::unique_ptr<FunctionApproximator>
  train(const Eigen::MatrixXd & inputs,
        const Eigen::MatrixXd & observations,
        const Eigen::MatrixXd & limits) const override;

  virtual std::unique_ptr<FunctionApproximator>
  train(const Eigen::MatrixXd & inputs,
        const Eigen::MatrixXd & observations,
        const Eigen::MatrixXd & limits,
        const FunctionApproximator & initial_fa) const override;

  /// Train several networks with different learning rates simultaneously and
  /// returns the best according to cross-validation loss
  DNNApproximator::network trainBestNN(const DNNApproximator::network & initial_network,
                                       const Eigen::MatrixXd & inputs,
                                       const Eigen::MatrixXd & outputs,
                                       bool reset_weights) const;

  /// Train the provided neural network with the given data and learning rate
  /// The final loss for the cross_validation set is placed in cv_loss
  void trainNN(const std::vector<tiny_dnn::vec_t> & training_inputs,
               const std::vector<tiny_dnn::vec_t> & training_outputs,
               const std::vector<tiny_dnn::vec_t> & cv_inputs,
               const std::vector<tiny_dnn::vec_t> & cv_outputs,
               double learning_rate,
               bool reset_weights,
               DNNApproximator::network * nn,
               double * cv_loss) const;

  virtual std::string getClassName() const override;
  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value & v, const std::string & dir_name) override;

private:
  /// Nb elements in each hidden layer
  DNNApproximator::Config dnn_config;

  /// Learning rates used (best DNN based on CV is chosen finally)
  std::vector<double> learning_rates;

  /// Number of minibatches
  int nb_minibatches;

  /// Number of train epochs
  int nb_train_epochs;

  /// Ratio of data used for cross validation ([0,1])
  double cv_ratio;

  /// Which loss function is used for training
  LossFunction loss;

  /// Verbosity of the trainer
  int verbose;
};

}
