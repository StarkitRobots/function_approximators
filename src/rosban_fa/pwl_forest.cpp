#include "rosban_fa/pwl_forest.h"

#include "rosban_regression_forests/algorithms/extra_trees.h"
#include "rosban_regression_forests/approximations/approximation_type.h"
#include "rosban_regression_forests/approximations/pwl_approximation.h"

using regression_forests::ApproximationType;
using regression_forests::PWLApproximation;
using regression_forests::ExtraTrees;
using regression_forests::TrainingSet;
using regression_forests::Forest;
using regression_forests::Tree;
using regression_forests::Node;

namespace rosban_fa
{

PWLForest::~PWLForest() {}

int PWLForest::getOutputDim() const
{
  return forests.size();
}

void PWLForest::train(const Eigen::MatrixXd & inputs,
                      const Eigen::MatrixXd & observations,
                      const Eigen::MatrixXd & limits)
{
  checkConsistency(inputs, observations, limits);
  ExtraTrees solver;
  solver.conf =  ExtraTrees::Config::generateAuto(limits,
                                                  observations.rows(),
                                                  ApproximationType::PWL);
  forests.clear();
  for (int output_dim = 0; output_dim < observations.cols(); output_dim++)
  {
    TrainingSet ts(inputs, observations.col(output_dim));
    forests.push_back(solver.solve(ts, limits));
  }
}

void PWLForest::predict(const Eigen::VectorXd & input,
                        Eigen::VectorXd & mean,
                        Eigen::MatrixXd & covar)
{
  int O = getOutputDim();
  mean = Eigen::VectorXd::Zero(O);
  covar = Eigen::MatrixXd::Zero(O,O);
  for (int output_dim = 0; output_dim < O; output_dim++)
  {
    mean(output_dim) = forests[output_dim]->getValue(input);
    covar(output_dim, output_dim) = forests[output_dim]->getVar(input);
  }
}

void PWLForest::gradient(const Eigen::VectorXd & input,
                         Eigen::VectorXd & gradient)
{
  (void) input;
  (void) gradient;
  throw std::runtime_error("PWLForest::gradients: not implemented");
}

void PWLForest::getMaximum(const Eigen::MatrixXd & limits,
                                 Eigen::VectorXd & input, double & output)
{
  check1DOutput("getMaximum");
  //TODO: as parameter
  //TODO: optional other way to compute max (gradient based)
  int max_action_tiles = 20000;
  std::unique_ptr<regression_forests::Tree> sub_tree;
  sub_tree = forests[0]->unifiedProjectedTree(limits, max_action_tiles);

  std::pair<double, Eigen::VectorXd> max_pair;
  max_pair = sub_tree->getMaxPair(limits);

  input = max_pair.second;
  output = max_pair.first;
}

std::string PWLForest::class_name() const
{
  return "pwl_forest_solver";
}

void PWLForest::to_xml(std::ostream &out) const
{
  (void) out;
  throw std::runtime_error("PWLForest::to_xml: unimplemented");
}

void PWLForest::from_xml(TiXmlNode *node)
{
  (void) node;
  throw std::runtime_error("PWLForest::from_xml: unimplemented");
}

}
