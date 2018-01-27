#pragma once

#include "rosban_fa/trainer.h"

#include "rosban_fa/dnn_approximator.h"

namespace rosban_fa
{

class DNNApproximatorTrainer : public Trainer {
public:
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

  void trainNN(DNNApproximator::network * nn,
               const Eigen::MatrixXd & input,
               const Eigen::MatrixXd & observations,
               bool reset_weights) const;

  virtual std::string getClassName() const override;
  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value & v, const std::string & dir_name) override;

private:
  /// Nb elements in each hidden layer
  std::vector<int> layer_units;

  /// Learning rate used
  double learning_rate;

  /// Number of minibatches
  int nb_minibatches;

  /// Number of train epochs
  int nb_train_epochs;

  /// Ratio of data used for cross validation ([0,1])
  double cv_ratio;

  /// Verbosity of the trainer
  int verbose;
};

}
