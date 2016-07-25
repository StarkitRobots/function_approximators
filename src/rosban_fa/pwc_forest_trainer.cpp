#include "rosban_fa/pwc_forest_trainer.h"

#include "rosban_fa/pwc_forest.h"

#include "rosban_regression_forests/approximations/approximation_type.h"
#include "rosban_regression_forests/algorithms/extra_trees.h"

#include "rosban_utils/xml_tools.h"

using regression_forests::ApproximationType;
using regression_forests::ExtraTrees;
using regression_forests::TrainingSet;

namespace rosban_fa
{

PWCForestTrainer::PWCForestTrainer()
  : max_action_tiles(2000)
{}

std::unique_ptr<FunctionApproximator>
PWCForestTrainer::train(const Eigen::MatrixXd & inputs,
                        const Eigen::MatrixXd & observations,
                        const Eigen::MatrixXd & limits) const
{
  checkConsistency(inputs, observations, limits);
  ExtraTrees solver;
  solver.conf =  ExtraTrees::Config::generateAuto(limits,
                                                  observations.rows(),
                                                  ApproximationType::PWC);

  std::unique_ptr<PWCForest::Forests> forests(new PWCForest::Forests());
  for (int  output_dim = 0; output_dim < observations.cols(); output_dim++)
  {
    TrainingSet ts(inputs, observations.col(output_dim));
    forests->push_back(solver.solve(ts, limits));
  }
  return std::unique_ptr<FunctionApproximator>(new PWCForest(std::move(forests), max_action_tiles));
}



std::string PWCForestTrainer::class_name() const
{
  return "PWCForestTrainer";
}

void PWCForestTrainer::to_xml(std::ostream &out) const
{
  rosban_utils::xml_tools::write<int>("max_action_tiles", max_action_tiles, out);
}

void PWCForestTrainer::from_xml(TiXmlNode *node)
{
  rosban_utils::xml_tools::try_read<int>(node, "max_action_tiles", max_action_tiles);
}

}
