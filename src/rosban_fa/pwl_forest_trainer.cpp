#include "rosban_fa/pwl_forest_trainer.cpp"

#include "rosban_fa/pwl_forest.cpp"

namespace rosban_fa
{

PWLForestTrainer::PWLForestTrainer()
  : max_action_tiles(2000);
{}

std::unique_ptr<FunctionApproximator>
PWLForest::train(const Eigen::MatrixXd & inputs,
                 const Eigen::MatrixXd & observations,
                 const Eigen::MatrixXd & limits) const
{
  checkConsistency(inputs, observations, limits);
  ExtraTrees solver;
  solver.conf =  ExtraTrees::Config::generateAuto(limits,
                                                  observations.rows(),
                                                  ApproximationType::PWL);

  std::unique_ptr<PWLForest::Forests> forests(new PWLForest::Forests());
  for (int  output_dim = 0; output_dim < observations.cols(); output_dim++)
  {
    TrainingSet ts(inputs, observations.col(output_dim));
    forests->push_back(solver.solve(ts, limits));
  }
  return std::unique_ptr<FunctionApproximator>(new PWLForest(forests, max_action_tiles));
}



std::string PWLForest::class_name() const
{
  return "PWLForestTrainer";
}

void PWLForest::to_xml(std::ostream &out) const
{
  rosban_utils::xml_tools::write<int>("max_action_tiles", max_action_tiles, out);
}

void PWLForest::from_xml(TiXmlNode *node)
{
  rosban_utils::xml_tools::tryRead<int>(node, "max_action_tiles", max_action_tiles);
}

}
