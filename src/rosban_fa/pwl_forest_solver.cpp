#include "regression_experiments/pwl_forest_solver.h"

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

namespace regression_experiments
{

PWLForestSolver::~PWLForestSolver() {}

void PWLForestSolver::solve(const Eigen::MatrixXd & inputs,
                            const Eigen::VectorXd & observations,
                            const Eigen::MatrixXd & limits)
{
  ExtraTrees solver;
  solver.conf =  ExtraTrees::Config::generateAuto(limits,
                                                  observations.rows(),
                                                  ApproximationType::PWL);
  TrainingSet ts(inputs, observations);

  forest = solver.solve(ts, limits);
}

void PWLForestSolver::predict(const Eigen::MatrixXd & inputs,
                             Eigen::VectorXd & means,
                             Eigen::VectorXd & vars)
{
  Eigen::VectorXd result(inputs.cols());
  means = Eigen::VectorXd::Zero(inputs.cols());
  vars = Eigen::VectorXd::Zero(inputs.cols());
  for (int point = 0; point < inputs.cols(); point++) {
    means(point) = forest->getValue(inputs.col(point));
    vars(point)  = forest->getVar(inputs.col(point));
  }
}

void PWLForestSolver::gradients(const Eigen::MatrixXd & inputs,
                               Eigen::MatrixXd & gradients)
{
  (void) inputs;
  (void) gradients;
  throw std::runtime_error("PWLForestSolver::gradients: not implemented");
}

void PWLForestSolver::getMaximum(const Eigen::MatrixXd & limits,
                                 Eigen::VectorXd & input, double & output)
{
  int max_action_tiles = 20000;
  std::unique_ptr<regression_forests::Tree> sub_tree;
  sub_tree = forest->unifiedProjectedTree(limits, max_action_tiles);

  std::pair<double, Eigen::VectorXd> max_pair;
  max_pair = sub_tree->getMaxPair(limits);

  input = max_pair.second;
  output = max_pair.first;
}

std::string PWLForestSolver::class_name() const
{
  return "pwl_forest_solver";
}

void PWLForestSolver::to_xml(std::ostream &out) const
{
  (void) out;
  throw std::runtime_error("PWLForestSolver::to_xml: unimplemented");
}

void PWLForestSolver::from_xml(TiXmlNode *node)
{
  (void) node;
  throw std::runtime_error("PWLForestSolver::from_xml: unimplemented");
}

}
