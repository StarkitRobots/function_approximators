#pragma once

#include "rosban_fa/trainer.h"

#include "rosban_regression_forests/approximations/approximation.h"
#include "rosban_regression_forests/core/forest.h"

namespace rosban_fa
{

class ForestTrainer : public Trainer
{
public:
  ForestTrainer();
  virtual ~ForestTrainer();

  /// Which type of approximation is used
  virtual regression_forests::Approximation::ID getApproximationID() const = 0;

  /// Update internal structure according to the provided samples
  /// Current method uses default configuration for extra-trees
  virtual std::unique_ptr<FunctionApproximator>
  train(const Eigen::MatrixXd & inputs,
        const Eigen::MatrixXd & observations,
        const Eigen::MatrixXd & limits) const override;

  virtual void to_xml(std::ostream &out) const override;
  virtual void from_xml(TiXmlNode *node) override;

  void setNbTrees(int nb_trees);

private:
  /// Maximal number of tiles used for getting max
  int max_action_tiles;
  /// Method used to get values from the forest
  regression_forests::Forest::AggregationMethod aggregation_method;
  /// Number of trees used for the approximator
  int nb_trees;
};

}
