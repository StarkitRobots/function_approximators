#pragma once

#include "rhoban_fa/trainer.h"

#include "rhoban_regression_forests/approximations/approximation.h"
#include "rhoban_regression_forests/core/forest.h"

namespace rhoban_fa
{

class ForestTrainer : public Trainer
{
public:
  ForestTrainer();
  virtual ~ForestTrainer();

  void setNbTrees(int nb_trees);

  /// Which type of approximation is used
  virtual regression_forests::Approximation::ID getApproximationID() const = 0;

  /// Update internal structure according to the provided samples
  /// Current method uses default configuration for extra-trees
  virtual std::unique_ptr<FunctionApproximator>
  train(const Eigen::MatrixXd & inputs,
        const Eigen::MatrixXd & observations,
        const Eigen::MatrixXd & limits) const override;

  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value & v, const std::string & dir_name) override;

private:
  /// Maximal number of tiles used for getting max
  int max_action_tiles;
  /// Method used to get values from the forest
  regression_forests::Forest::AggregationMethod aggregation_method;
  /// Number of trees used for the approximator
  int nb_trees;
};

}
