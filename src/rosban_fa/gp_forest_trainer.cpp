#include "rosban_fa/gp_forest_trainer.h"

#include "rosban_fa/gp_forest.h"

#include "rosban_regression_forests/approximations/approximation_type.h"
#include "rosban_regression_forests/algorithms/extra_trees.h"

using regression_forests::ApproximationType;
using regression_forests::ExtraTrees;
using regression_forests::TrainingSet;

namespace rosban_fa
{

GPForestTrainer::GPForestTrainer(Type t)
  : type(t), nb_threads(1)
{
  autotune_conf.nb_trials = 2;
  autotune_conf.rprop_conf->max_iterations = 50;
  autotune_conf.rprop_conf->epsilon = std::pow(10,-6);
  autotune_conf.rprop_conf->tuning_space = rosban_gp::RProp::TuningSpace::Log;
  ga_conf.nb_trials = 10;
  ga_conf.rprop_conf->max_iterations = 1000;
  ga_conf.rprop_conf->epsilon = std::pow(10,-6);
}

std::unique_ptr<FunctionApproximator>
GPForestTrainer::train(const Eigen::MatrixXd & inputs,
                       const Eigen::MatrixXd & observations,
                       const Eigen::MatrixXd & limits) const
{
  checkConsistency(inputs, observations, limits);

  int nb_samples = observations.rows();

  ExtraTrees solver;
  solver.conf =  ExtraTrees::Config::generateAuto(limits, nb_samples,
                                                  ApproximationType::GP);

  // Updating nmin with respect to type
  switch(type)
  {
    case GPForestTrainer::Type::SQRT:
      solver.conf.n_min = std::sqrt(nb_samples);
      break;
    case GPForestTrainer::Type::CURT:
      solver.conf.n_min = std::pow(nb_samples, 1.0 / 3);
      break;
    case GPForestTrainer::Type::LOG2:
      solver.conf.n_min = std::log2(nb_samples);
      break;
  }
  solver.conf.nb_threads = nb_threads;
  solver.conf.gp_conf = autotune_conf;

std::unique_ptr<GPForest::Forests> forests(new GPForest::Forests());
  for (int output_dim = 0; output_dim < observations.cols(); output_dim++)
  {
    TrainingSet ts(inputs, observations.col(output_dim));
    forests->push_back(solver.solve(ts, limits));
  }
  return std::unique_ptr<FunctionApproximator>(new GPForest(std::move(forests), ga_conf));
}

std::string GPForestTrainer::class_name() const
{
  return "GPForestTrainer";
}

void GPForestTrainer::to_xml(std::ostream &out) const
{
  rosban_utils::xml_tools::write<std::string>("type", to_string(type), out);
  rosban_utils::xml_tools::write<int>("nb_threads", nb_threads, out);
  autotune_conf.write("autotune_conf", out);
  ga_conf.write("ga_conf", out);
}

void GPForestTrainer::from_xml(TiXmlNode *node)
{
  std::string type_str;
  rosban_utils::xml_tools::try_read<std::string>(node, "type", type_str);
  if (type_str.size() != 0) type = loadType(type_str);
  rosban_utils::xml_tools::try_read<int>(node, "nb_threads", nb_threads);
  autotune_conf.tryRead(node, "auto_tune_conf");
  ga_conf.tryRead(node, "ga_conf");
}

GPForestTrainer::Type GPForestTrainer::loadType(const std::string &s)
{
  if (s == "SQRT") return GPForestTrainer::Type::SQRT;
  if (s == "CURT") return GPForestTrainer::Type::CURT;
  if (s == "LOG2") return GPForestTrainer::Type::LOG2;
  throw std::runtime_error("Unknown GPForestTrainer::Type: '" + s + "'");
}

std::string to_string(GPForestTrainer::Type type)
{
  switch(type)
  {
    case GPForestTrainer::Type::SQRT: return "SQRT";
    case GPForestTrainer::Type::CURT: return "CURT";
    case GPForestTrainer::Type::LOG2: return "LOG2";
  }
  throw std::runtime_error("GPForestTrainer::Type unknown in to_string");
}

}
