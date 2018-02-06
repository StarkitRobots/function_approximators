#include "rhoban_fa/gp_forest_trainer.h"

#include "rhoban_fa/gp_forest.h"

#include "rhoban_regression_forests/algorithms/extra_trees.h"

#include "rhoban_utils/io_tools.h"

using regression_forests::Approximation;
using regression_forests::ExtraTrees;
using regression_forests::TrainingSet;

namespace rhoban_fa
{

GPForestTrainer::GPForestTrainer(Type t)
  : type(t)
{
  autotune_conf.nb_trials = 2;
  autotune_conf.rprop_conf->max_iterations = 50;
  autotune_conf.rprop_conf->epsilon = std::pow(10,-6);
  autotune_conf.rprop_conf->tuning_space = rhoban_gp::RProp::TuningSpace::Log;
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
                                                  Approximation::ID::GP);

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

std::string GPForestTrainer::getClassName() const
{
  return "GPForestTrainer";
}

Json::Value GPForestTrainer::toJson() const
{
  Json::Value v = Trainer::toJson();
  v["type"] = to_string(type);
  v["autotune_conf"] = autotune_conf.toJson();
  v["ga_conf"] = ga_conf.toJson();
  return v;
}

void GPForestTrainer::fromJson(const Json::Value & v, const std::string & dir_name)
{
  Trainer::fromJson(v, dir_name);
  std::string type_str;
  rhoban_utils::tryRead(v, "type", &type_str);
  if (type_str.size() != 0) type = loadType(type_str);
  autotune_conf.tryRead(v, "auto_tune_conf", dir_name);
  ga_conf.tryRead(v, "ga_conf", dir_name);
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
