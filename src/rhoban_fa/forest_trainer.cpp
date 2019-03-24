#include "rhoban_fa/forest_trainer.h"

#include "rhoban_fa/pwc_forest.h"
#include "rhoban_fa/pwl_forest.h"

#include "rhoban_regression_forests/algorithms/extra_trees.h"

using regression_forests::Approximation;
using regression_forests::ExtraTrees;
using regression_forests::Forest;
using regression_forests::TrainingSet;

namespace rhoban_fa
{
ForestTrainer::ForestTrainer()
  : max_action_tiles(2000), aggregation_method(Forest::AggregationMethod::All), nb_trees(25)
{
}

ForestTrainer::~ForestTrainer()
{
}

void ForestTrainer::setNbTrees(int new_nb_trees)
{
  nb_trees = new_nb_trees;
}

std::unique_ptr<FunctionApproximator> ForestTrainer::train(const Eigen::MatrixXd& inputs,
                                                           const Eigen::MatrixXd& observations,
                                                           const Eigen::MatrixXd& limits) const
{
  checkConsistency(inputs, observations, limits);
  ExtraTrees solver;
  solver.conf = ExtraTrees::Config::generateAuto(limits, observations.rows(), getApproximationID());
  solver.conf.nb_threads = nb_threads;
  solver.conf.nb_trees = nb_trees;

  std::unique_ptr<ForestApproximator::Forests> forests(new ForestApproximator::Forests());
  for (int output_dim = 0; output_dim < observations.cols(); output_dim++)
  {
    TrainingSet ts(inputs, observations.col(output_dim));
    forests->push_back(solver.solve(ts, limits));
  }
  std::unique_ptr<ForestApproximator> result;
  switch (getApproximationID())
  {
    case Approximation::ID::PWC:
      result.reset(new PWCForest);
      break;
    case Approximation::ID::PWL:
      result.reset(new PWLForest);
      break;
    default:
      throw std::runtime_error("Unhandled type in forest_trainer");
  }
  result->setForests(std::move(forests));
  result->setMaxActionTiles(max_action_tiles);
  result->setAggregationMethod(aggregation_method);
  return std::move(result);
}

Json::Value ForestTrainer::toJson() const
{
  Json::Value v = Trainer::toJson();
  v["max_action_tiles"] = max_action_tiles;
  v["nb_trees"] = nb_trees;
  v["am_str"] = regression_forests::aggregationMethod2Str(aggregation_method);
  return v;
}

void ForestTrainer::fromJson(const Json::Value& v, const std::string& dir_name)
{
  Trainer::fromJson(v, dir_name);
  std::string am_str;
  rhoban_utils::tryRead(v, "max_action_tiles", &max_action_tiles);
  rhoban_utils::tryRead(v, "nb_trees", &nb_trees);
  rhoban_utils::tryRead(v, "aggregation_method", &am_str);
  if (am_str != "")
    aggregation_method = regression_forests::loadAggregationMethod(am_str);
}

}  // namespace rhoban_fa
