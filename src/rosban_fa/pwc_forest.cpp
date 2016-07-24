#include "rosban_fa/pwc_forest.h"

#include "rosban_regression_forests/algorithms/extra_trees.h"
#include "rosban_regression_forests/approximations/approximation_type.h"
#include "rosban_regression_forests/approximations/pwc_approximation.h"

using regression_forests::ApproximationType;
using regression_forests::PWCApproximation;
using regression_forests::ExtraTrees;
using regression_forests::TrainingSet;
using regression_forests::Forest;
using regression_forests::Tree;
using regression_forests::Node;

namespace rosban_fa
{

PWCForest::~PWCForest() {}

int PWCForest::getOutputDim() const
{
  return forests.size();
}

void PWCForest::train(const Eigen::MatrixXd & inputs,
                      const Eigen::MatrixXd & observations,
                      const Eigen::MatrixXd & limits)
{
  checkConsistency(inputs, observations, limits);
  ExtraTrees solver;
  solver.conf =  ExtraTrees::Config::generateAuto(limits,
                                                  observations.rows(),
                                                  ApproximationType::PWC);
  forests.clear();
  for (int  output_dim = 0; output_dim < observations.cols(); output_dim++)
  {
    TrainingSet ts(inputs, observations.col(output_dim));
    forests.push_back(solver.solve(ts, limits));
  }
}

void PWCForest::predict(const Eigen::VectorXd & input,
                        Eigen::VectorXd & mean,
                        Eigen::MatrixXd & covar)
{
  int O = getOutputDim();
  mean = Eigen::VectorXd::Zero(O);
  Eigen::VectorXd vars = Eigen::VectorXd::Zero(O);
  for (int output_dim = 0; output_dim < O; output_dim++)
  {
    mean(output_dim) = forests[output_dim]->getValue(input);
    vars(output_dim) = forests[output_dim]->getVar(input);
  }
  covar = Eigen::MatrixXd::Identity(O,O) * vars;
}

void PWCForest::gradient(const Eigen::VectorXd & input,
                         Eigen::VectorXd & gradient)
{
  (void) input;(void) gradient;
  throw std::runtime_error("PWCForest::gradients: not implemented");
}

void PWCForest::getMaximum(const Eigen::MatrixXd & limits,
                           Eigen::VectorXd & input,
                           double & output)
{
  check1DOutput("getMaximum");
  //TODO max_action_tiles as parameter
  //TODO eventually use gradient ascent as optional part
  int max_action_tiles = 2000;
  std::unique_ptr<regression_forests::Tree> sub_tree;
  sub_tree = forests[0]->unifiedProjectedTree(limits, max_action_tiles);

  std::pair<double, Eigen::VectorXd> max_pair;
  max_pair = sub_tree->getMaxPair(limits);

  input = max_pair.second;
  output = max_pair.first;
}

std::string PWCForest::class_name() const
{
  return "pwc_forest";
}

void PWCForest::to_xml(std::ostream &out) const
{
  (void) out;
  throw std::runtime_error("PWCForest::to_xml: unimplemented");
}

void PWCForest::from_xml(TiXmlNode *node)
{
  (void) node;
  throw std::runtime_error("PWCForest::from_xml: unimplemented");
}

}
