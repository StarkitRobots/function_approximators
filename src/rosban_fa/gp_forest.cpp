#include "rosban_fa/gp_forest.h"

#include "rosban_regression_forests/algorithms/extra_trees.h"
#include "rosban_regression_forests/approximations/approximation_type.h"
#include "rosban_regression_forests/approximations/gp_approximation.h"

#include "rosban_gp/auto_tuning.h"
#include "rosban_gp/tools.h"
#include "rosban_gp/core/gaussian_process.h"

#include "rosban_utils/serializable.h"

#include "rosban_random/tools.h"

using regression_forests::ApproximationType;
using regression_forests::GPApproximation;
using regression_forests::ExtraTrees;
using regression_forests::TrainingSet;
using regression_forests::Forest;
using regression_forests::Tree;
using regression_forests::Node;

using rosban_gp::GaussianProcess;

namespace rosban_fa
{

GPForest::GPForest(Type t)
  : type(t), nb_threads(1)
{
}
GPForest::~GPForest() {}

int GPForest::getOutputDim() const
{
  return forests.size();
}

void GPForest::train(const Eigen::MatrixXd & inputs,
                     const Eigen::MatrixXd & observations,
                     const Eigen::MatrixXd & limits)
{
  int nb_samples = observations.rows();

  ExtraTrees solver;
  solver.conf =  ExtraTrees::Config::generateAuto(limits, nb_samples,
                                                  ApproximationType::GP);

  // Updating nmin with respect to type
  switch(type)
  {
    case GPForest::Type::SQRT:
      solver.conf.n_min = std::sqrt(nb_samples);
      break;
    case GPForest::Type::CURT:
      solver.conf.n_min = std::pow(nb_samples, 1.0 / 3);
      break;
    case GPForest::Type::LOG2:
      solver.conf.n_min = std::log2(nb_samples);
      break;
  }
  solver.conf.nb_threads = nb_threads;
  solver.conf.gp_conf = approximation_conf;

  forests.clear();
  for (int output_dim = 0; output_dim < observations.cols(); output_dim++)
  {
    TrainingSet ts(inputs, observations.col(output_dim));

    // Solve problem
    forests.push_back(solver.solve(ts, limits));
  }
}

void GPForest::predict(const Eigen::VectorXd & input,
                             Eigen::VectorXd & mean,
                             Eigen::MatrixXd & covar)
{
  Eigen::VectorXd vars(forests.size());
  mean = Eigen::VectorXd(forests.size());
  for (size_t output_dim = 0; output_dim < forests.size(); output_dim++) {
    // TODO: change design to make something 'cleaner'
    // Retrieving all the gaussian processes at the given point
    std::vector<GaussianProcess> gps;
    for (size_t tree_id = 0; tree_id < forests[output_dim]->nbTrees(); tree_id++) {
      const Tree & tree = forests[output_dim]->getTree(tree_id);
      const Node * leaf = tree.root->getLeaf(input);
      const GPApproximation * gp_approximation = dynamic_cast<const GPApproximation *>(leaf->a);
      if (gp_approximation == nullptr) {
        throw std::runtime_error("Found an approximation which is not a gaussian process");
      }
      gps.push_back(gp_approximation->gp);
    }
    // Averaging gaussian processes
    double tmp_mean, tmp_var;
    rosban_gp::getDistribParameters(input, gps, tmp_mean, tmp_var);
    mean(output_dim) = tmp_mean;
    vars(output_dim) = tmp_var;
  }
  covar = Eigen::MatrixXd::Identity(forests.size(), forests.size()) * vars;
}

void GPForest::debugPrediction(const Eigen::VectorXd & input, std::ostream & out)
{
  for (size_t output_dim = 0; output_dim < forests.size(); output_dim++)
  {
    out << "### Debug along dimension " << output_dim << std::endl;
    // Retrieving gaussian processes at the given point
    std::vector<GaussianProcess> gps;
    for (size_t tree_id = 0; tree_id < forests[output_dim]->nbTrees(); tree_id++) {
      const Tree & tree = forests[output_dim]->getTree(tree_id);
      const Node * leaf = tree.root->getLeaf(input);
      const GPApproximation * gp_approximation = dynamic_cast<const GPApproximation *>(leaf->a);
      if (gp_approximation == nullptr) {
        throw std::runtime_error("Found an approximation which is not a gaussian process");
      }
      gps.push_back(gp_approximation->gp);
    }
    // Averaging gaussian processes
    double mean, var;
    rosban_gp::getDistribParameters(input, gps, mean, var, &out);
  }
}


void GPForest::gradient(const Eigen::VectorXd & input,
                        Eigen::VectorXd & gradient)
{
  check1DOutput("gradient");
  gradient = Eigen::VectorXd::Zero(input.rows());
  double total_weight = 0;
  for (size_t tree_id = 0; tree_id < forests[0]->nbTrees(); tree_id++) {
    const Tree & tree = forests[0]->getTree(tree_id);
    const Node * leaf = tree.root->getLeaf(input);
    const GPApproximation * gp_approximation = dynamic_cast<const GPApproximation *>(leaf->a);
    if (gp_approximation == nullptr) {
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
                          double & output)
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
  best_guess = rosban_gp::RandomizedRProp::run(gradient_func, scoring_func, limits,
                                               find_max_conf);
  input = best_guess;
  output = scoring_func(best_guess);
}

std::string GPForest::class_name() const
{
  return "gp_forest";
}

void GPForest::to_xml(std::ostream &out) const
{
  rosban_utils::xml_tools::write<std::string>("type", to_string(type), out);
  rosban_utils::xml_tools::write<int>("nb_threads", nb_threads, out);
  approximation_conf.write("approximation_conf", out);
  find_max_conf.write("find_max_conf", out);
}

void GPForest::from_xml(TiXmlNode *node)
{
  std::string type_str;
  rosban_utils::xml_tools::try_read<std::string>(node, "type", type_str);
  if (type_str.size() != 0) type = loadType(type_str);
  rosban_utils::xml_tools::try_read<int>(node, "nb_threads", nb_threads);
  approximation_conf.tryRead(node, "approximation_conf");
  find_max_conf.tryRead(node, "find_max_conf");
}

GPForest::Type loadType(const std::string &s)
{
  if (s == "SQRT") return GPForest::Type::SQRT;
  if (s == "CURT") return GPForest::Type::CURT;
  if (s == "LOG2") return GPForest::Type::LOG2;
  throw std::runtime_error("Unknown GPForest::Type: '" + s + "'");
}

std::string to_string(GPForest::Type type)
{
  switch(type)
  {
    case GPForest::Type::SQRT: return "SQRT";
    case GPForest::Type::CURT: return "CURT";
    case GPForest::Type::LOG2: return "LOG2";
  }
  throw std::runtime_error("GPForest::Type unknown in to_string");
}

}
