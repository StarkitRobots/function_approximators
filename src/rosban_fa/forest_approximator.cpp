#include "rosban_fa/forest_approximator.h"

#include "rosban_regression_forests/algorithms/extra_trees.h"

#include "rhoban_utils/io_tools.h"

using regression_forests::Forest;
using regression_forests::Tree;
using regression_forests::Node;

namespace rosban_fa
{

ForestApproximator::ForestApproximator()
  : aggregation_method(Forest::AggregationMethod::All)
{}

ForestApproximator::ForestApproximator(std::unique_ptr<Forests> forests_,
                     int max_action_tiles_)
  : forests(std::move(forests_)), max_action_tiles(max_action_tiles_),
    aggregation_method(Forest::AggregationMethod::All)
{}

ForestApproximator::~ForestApproximator() {}

std::unique_ptr<FunctionApproximator> ForestApproximator::clone() const {
  throw std::logic_error("ForestApproximator::clone: not implemented");
}

void ForestApproximator::setForests(std::unique_ptr<Forests> new_forests)
{
  forests = std::move(new_forests);
}

void ForestApproximator::setMaxActionTiles(int new_mat)
{
  max_action_tiles = new_mat;
}

void ForestApproximator::setAggregationMethod(Forest::AggregationMethod new_am)
{
  aggregation_method = new_am;
}

int ForestApproximator::getOutputDim() const
{
  if (!forests) return 0;
  return forests->size();
}

void ForestApproximator::predict(const Eigen::VectorXd & input,
                        Eigen::VectorXd & mean,
                        Eigen::MatrixXd & covar) const
{
  int O = getOutputDim();
  mean = Eigen::VectorXd::Zero(O);
  covar = Eigen::MatrixXd::Zero(O,O);
  Eigen::VectorXd vars = Eigen::VectorXd::Zero(O);
  for (int output_dim = 0; output_dim < O; output_dim++)
  {
    mean(output_dim) = (*forests)[output_dim]->getValue(input, aggregation_method);
    covar(output_dim, output_dim) = (*forests)[output_dim]->getVar(input, aggregation_method);
  }
}

void ForestApproximator::gradient(const Eigen::VectorXd & input,
                                  Eigen::VectorXd & gradient) const
{
  check1DOutput("ForestApproximator");
  gradient = (*forests)[0]->getGradient(input);
}

void ForestApproximator::getMaximum(const Eigen::MatrixXd & limits,
                           Eigen::VectorXd & input,
                           double & output) const
{
  check1DOutput("getMaximum");
  std::unique_ptr<regression_forests::Tree> sub_tree;
  sub_tree = (*forests)[0]->unifiedProjectedTree(limits, max_action_tiles);

  std::pair<double, Eigen::VectorXd> max_pair;
  max_pair = sub_tree->getMaxPair(limits);

  input = max_pair.second;
  output = max_pair.first;
}

int ForestApproximator::getClassID() const
{
  return FunctionApproximator::ForestApproximator;
}

int ForestApproximator::writeInternal(std::ostream & out) const
{
  int bytes_written = 0;
  bytes_written += rhoban_utils::write<int>(out, getOutputDim());
  for (int dim = 0; dim < getOutputDim(); dim++) {
    bytes_written += (*forests)[dim]->writeInternal(out);
  }
  bytes_written += rhoban_utils::write<int>(out, max_action_tiles);
  return bytes_written;
}

int ForestApproximator::read(std::istream & in)
{
  // First clear existing data
  if (forests) forests.release();
  forests = std::unique_ptr<Forests>(new Forests());
  // Then read
  int bytes_read = 0;
  int output_dim;
  bytes_read += rhoban_utils::read<int>(in, &output_dim);
  for (int dim = 0; dim < output_dim; dim++) {
    std::unique_ptr<Forest> ptr(new Forest);
    bytes_read += ptr->read(in);
    forests->push_back(std::move(ptr));
  }
  bytes_read += rhoban_utils::read<int>(in, &max_action_tiles);
  return bytes_read;
}

}
