#include "rosban_fa/pwl_forest_trainer.h"

#include "rosban_fa/pwl_forest.h"

#include "rosban_regression_forests/algorithms/extra_trees.h"

using regression_forests::Approximation;
using regression_forests::ExtraTrees;
using regression_forests::TrainingSet;

namespace rosban_fa
{

PWLForestTrainer::PWLForestTrainer()
  : max_action_tiles(2000)
{}

std::unique_ptr<FunctionApproximator>
PWLForestTrainer::train(const Eigen::MatrixXd & inputs,
                        const Eigen::MatrixXd & observations,
                        const Eigen::MatrixXd & limits) const
{
  checkConsistency(inputs, observations, limits);
  ExtraTrees solver;
  solver.conf =  ExtraTrees::Config::generateAuto(limits,
                                                  observations.rows(),
                                                  Approximation::ID::PWL);

  std::unique_ptr<PWLForest::Forests> forests(new PWLForest::Forests());
  for (int  output_dim = 0; output_dim < observations.cols(); output_dim++)
  {
    TrainingSet ts(inputs, observations.col(output_dim));
    forests->push_back(solver.solve(ts, limits));
  }
  return std::unique_ptr<FunctionApproximator>(new PWLForest(std::move(forests), max_action_tiles));
}

std::string PWLForestTrainer::class_name() const
{
  return "PWLForestTrainer";
}

void PWLForestTrainer::to_xml(std::ostream &out) const
{
  rosban_utils::xml_tools::write<int>("max_action_tiles", max_action_tiles, out);
}

void PWLForestTrainer::from_xml(TiXmlNode *node)
{
  rosban_utils::xml_tools::try_read<int>(node, "max_action_tiles", max_action_tiles);
}

}
