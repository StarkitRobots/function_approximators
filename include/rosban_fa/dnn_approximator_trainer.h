#pragma once

#include "rosban_fa/trainer.h"

namespace rosban_fa
{

class DNNApproximatorTrainer : public Trainer {
public:
  DNNApproximatorTrainer();

  virtual std::unique_ptr<FunctionApproximator>
  train(const Eigen::MatrixXd & inputs,
        const Eigen::MatrixXd & observations,
        const Eigen::MatrixXd & limits) const override;

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
