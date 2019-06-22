#include "starkit_fa/gp_trainer.h"

#include "starkit_fa/gp.h"

#include "starkit_gp/core/squared_exponential.h"

using starkit_gp::CovarianceFunction;
using starkit_gp::GaussianProcess;
using starkit_gp::SquaredExponential;

namespace starkit_fa
{
GPTrainer::GPTrainer()
{
  autotune_conf.nb_trials = 2;
  autotune_conf.rprop_conf->max_iterations = 50;
  autotune_conf.rprop_conf->epsilon = std::pow(10, -6);
  autotune_conf.rprop_conf->tuning_space = starkit_gp::RProp::TuningSpace::Log;
  ga_conf.nb_trials = 10;
  ga_conf.rprop_conf->max_iterations = 1000;
  ga_conf.rprop_conf->epsilon = std::pow(10, -6);
}

std::unique_ptr<FunctionApproximator> GPTrainer::train(const Eigen::MatrixXd& inputs,
                                                       const Eigen::MatrixXd& observations,
                                                       const Eigen::MatrixXd& limits) const
{
  checkConsistency(inputs, observations, limits);
  std::unique_ptr<std::vector<starkit_gp::GaussianProcess>> gps(new std::vector<starkit_gp::GaussianProcess>());
  for (int output_dim = 0; output_dim < observations.cols(); output_dim++)
  {
    std::unique_ptr<CovarianceFunction> cov_func(new SquaredExponential(inputs.rows()));
    gps->push_back(GaussianProcess(inputs, observations.col(output_dim), std::move(cov_func)));
    (*gps)[output_dim].autoTune(autotune_conf);
  }
  return std::unique_ptr<FunctionApproximator>(new GP(std::move(gps), ga_conf));
}

std::string GPTrainer::getClassName() const
{
  return "GPTrainer";
}

Json::Value GPTrainer::toJson() const
{
  Json::Value v = Trainer::toJson();
  v["autotune_conf"] = autotune_conf.toJson();
  v["ga_conf"] = ga_conf.toJson();
  return v;
}

void GPTrainer::fromJson(const Json::Value& v, const std::string& dir_name)
{
  Trainer::fromJson(v, dir_name);
  autotune_conf.tryRead(v, "auto_tune_conf");
  ga_conf.tryRead(v, "ga_conf");
}

}  // namespace starkit_fa
