#include "rosban_fa/forest_trainer.h"

#include "rosban_fa/pwc_forest.h"
#include "rosban_fa/pwl_forest.h"

#include "rosban_regression_forests/algorithms/extra_trees.h"

using regression_forests::Approximation;
using regression_forests::ExtraTrees;
using regression_forests::Forest;
using regression_forests::TrainingSet;

namespace rosban_fa
{

ForestTrainer::ForestTrainer()
  : max_action_tiles(2000),
    aggregation_method(Forest::AggregationMethod::All),
    nb_trees(25)
{}

ForestTrainer::~ForestTrainer() {}

std::unique_ptr<FunctionApproximator>
ForestTrainer::train(const Eigen::MatrixXd & inputs,
                        const Eigen::MatrixXd & observations,
                        const Eigen::MatrixXd & limits) const
{
  checkConsistency(inputs, observations, limits);
  ExtraTrees solver;
  solver.conf =  ExtraTrees::Config::generateAuto(limits,
                                                  observations.rows(),
                                                  getApproximationID());
  solver.conf.nb_threads = nb_threads;
  solver.conf.nb_trees = nb_trees;

  std::unique_ptr<ForestApproximator::Forests> forests(new ForestApproximator::Forests());
  for (int  output_dim = 0; output_dim < observations.cols(); output_dim++)
  {
    TrainingSet ts(inputs, observations.col(output_dim));
    forests->push_back(solver.solve(ts, limits));
  }
  std::unique_ptr<ForestApproximator> result;
  switch(getApproximationID())
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

void ForestTrainer::to_xml(std::ostream &out) const
{
  rosban_utils::xml_tools::write<int>("max_action_tiles", max_action_tiles, out);
  std::string am_str = regression_forests::aggregationMethod2Str(aggregation_method);
  rosban_utils::xml_tools::write<std::string>("aggregation_method", am_str, out);
  rosban_utils::xml_tools::write<int>("nb_trees", nb_trees, out);
}

void ForestTrainer::from_xml(TiXmlNode *node)
{
  rosban_utils::xml_tools::try_read<int>(node, "max_action_tiles", max_action_tiles);
  std::string am_str;
  rosban_utils::xml_tools::try_read<std::string>(node, "aggregation_method", am_str);
  if (am_str != "") aggregation_method = regression_forests::loadAggregationMethod(am_str);
  rosban_utils::xml_tools::try_read<int>(node, "nb_trees", nb_trees);
}

void ForestTrainer::setNbTrees(int new_nb_trees) {
  nb_trees = new_nb_trees;
}

}
