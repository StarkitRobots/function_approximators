#include "rhoban_fa/gp_forest.h"

#include "rhoban_regression_forests/approximations/gp_approximation.h"

#include "rhoban_gp/auto_tuning.h"
#include "rhoban_gp/tools.h"
#include "rhoban_gp/core/gaussian_process.h"

#include "rhoban_random/tools.h"

using regression_forests::GPApproximation;
using regression_forests::Forest;
using regression_forests::Tree;
using regression_forests::Node;

using rhoban_gp::GaussianProcess;

namespace rhoban_fa
{

GPForest::GPForest() {}

GPForest::GPForest(std::unique_ptr<Forests> forests_,
                   const rhoban_gp::RandomizedRProp::Config & ga_conf_)
  : ForestApproximator(std::move(forests_), 0), ga_conf(ga_conf_)
{
}

GPForest::~GPForest() {}

std::unique_ptr<FunctionApproximator> GPForest::clone() const {
  throw std::logic_error("GPForest::clone: not implemented");
}

void GPForest::predict(const Eigen::VectorXd & input,
                             Eigen::VectorXd & mean,
                             Eigen::MatrixXd & covar) const
{
  int O = getOutputDim();
  mean = Eigen::VectorXd(O);
  covar = Eigen::MatrixXd::Zero(O,O);
  for (int output_dim = 0; output_dim < O; output_dim++) {
    // TODO: change design to make something 'cleaner'
    // Retrieving all the gaussian processes at the given point
    std::vector<GaussianProcess> gps;
    for (size_t tree_id = 0; tree_id < (*forests)[output_dim]->nbTrees(); tree_id++) {
      const Tree & tree = (*forests)[output_dim]->getTree(tree_id);
      const Node * leaf = tree.root->getLeaf(input);
      std::shared_ptr<const GPApproximation> gp_approximation;
      gp_approximation = std::dynamic_pointer_cast<const GPApproximation>(leaf->a);
      if (!gp_approximation) {
        throw std::runtime_error("Found an approximation which is not a gaussian process");
      }
      gps.push_back(gp_approximation->gp);
    }
    // Averaging gaussian processes
    double tmp_mean, tmp_var;
    rhoban_gp::getDistribParameters(input, gps, tmp_mean, tmp_var);
    mean(output_dim) = tmp_mean;
    covar(output_dim, output_dim) = tmp_var;
  }
}

void GPForest::debugPrediction(const Eigen::VectorXd & input, std::ostream & out) const
{
  for (int output_dim = 0; output_dim < getOutputDim(); output_dim++)
  {
    out << "### Debug along dimension " << output_dim << std::endl;
    // Retrieving gaussian processes at the given point
    std::vector<GaussianProcess> gps;
    for (size_t tree_id = 0; tree_id < (*forests)[output_dim]->nbTrees(); tree_id++) {
      const Tree & tree = (*forests)[output_dim]->getTree(tree_id);
      const Node * leaf = tree.root->getLeaf(input);
      std::shared_ptr<const GPApproximation> gp_approximation;
      gp_approximation = std::dynamic_pointer_cast<const GPApproximation>(leaf->a);
      if (!gp_approximation) {
        throw std::runtime_error("Found an approximation which is not a gaussian process");
      }
      gps.push_back(gp_approximation->gp);
    }
    // Averaging gaussian processes
    double mean, var;
    rhoban_gp::getDistribParameters(input, gps, mean, var, &out);
  }
}


void GPForest::gradient(const Eigen::VectorXd & input,
                        Eigen::VectorXd & gradient) const
{
  check1DOutput("gradient");
  gradient = Eigen::VectorXd::Zero(input.rows());
  double total_weight = 0;
  for (size_t tree_id = 0; tree_id < (*forests)[0]->nbTrees(); tree_id++) {
    const Tree & tree = (*forests)[0]->getTree(tree_id);
    const Node * leaf = tree.root->getLeaf(input);
    std::shared_ptr<const GPApproximation> gp_approximation;
    gp_approximation = std::dynamic_pointer_cast<const GPApproximation>(leaf->a);
    if (!gp_approximation) {
      throw std::runtime_error("Found an approximation which is not a gaussian process");
    }
    double var = gp_approximation->gp.getVariance(input);
    if (var == 0) {
      // DIRTY HACK
      var = std::pow(10,-20);//Avoiding to get a value of 0 for var
    }
    double weight = 1.0 / var;
    gradient += gp_approximation->gp.getGradient(input) * weight;
    total_weight += weight;
  }
  gradient /= total_weight;
}

void GPForest::getMaximum(const Eigen::MatrixXd & limits,
                          Eigen::VectorXd & input,
                          double & output) const
{
  check1DOutput("getMaximum");
  // Preparing functions
  std::function<Eigen::VectorXd(const Eigen::VectorXd)> gradient_func;
  gradient_func = [this](const Eigen::VectorXd & guess)
    {
      Eigen::VectorXd gradient;
      this->gradient(guess, gradient);
      return gradient;
    };
  std::function<double(const Eigen::VectorXd)> scoring_func;
  scoring_func = [this](const Eigen::VectorXd & guess)
    {
      Eigen::VectorXd value;
      Eigen::MatrixXd var;
      //TODO investigate on why predict(Vector, double, double) is not accepted
      this->predict(guess, value, var);
      return value(0);
    };
  // Performing multiple rProp and conserving the best candidate
  Eigen::VectorXd best_guess;
  best_guess = rhoban_gp::RandomizedRProp::run(gradient_func, scoring_func, limits,
                                               ga_conf);
  input = best_guess;
  output = scoring_func(best_guess);
}

int GPForest::getClassID() const
{
  return FunctionApproximator::GPForest;
}

int GPForest::writeInternal(std::ostream & out) const
{
  int bytes_written = 0;
  bytes_written += ForestApproximator::writeInternal(out);
  bytes_written += ga_conf.write(out);
  return bytes_written;
}

int GPForest::read(std::istream & in)
{
  // Then read
  int bytes_read = 0;
  bytes_read += ForestApproximator::read(in);
  bytes_read += ga_conf.read(in);
  return bytes_read;
}

}
