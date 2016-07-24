#include "rosban_fa/gp_trainer.h"

#include "rosban_fa/gp.h"

#include "rosban_gp/core/squared_exponential.h"

using rosban_gp::CovarianceFunction;
using rosban_gp::GaussianProcess;
using rosban_gp::SquaredExponential;

namespace rosban_fa
{

GPTrainer::GPTrainer()
{
  autotune_conf.nb_trials = 2;
  autotune_conf.rprop_conf->max_iterations = 50;
  autotune_conf.rprop_conf->epsilon = std::pow(10,-6);
  ga_conf.nb_trials = 10;
  ga_conf.rprop_conf->max_iterations = 1000;
  ga_conf.rprop_conf->epsilon = std::pow(10,-6);
}

std::unique_ptr<FunctionApproximator> GPTrainer::train(const Eigen::MatrixXd & inputs,
                                                       const Eigen::MatrixXd & observations,
                                                       const Eigen::MatrixXd & limits) const
{
  checkConsistency(inputs, observations, limits);
  std::unique_ptr<std::vector<rosban_gp::GaussianProcess>> gps(new std::vector<rosban_gp::GaussianProcess>());
  for (int output_dim = 0; output_dim < observations.cols(); output_dim++)
  {
    std::unique_ptr<CovarianceFunction> cov_func(new SquaredExponential(inputs.rows()));
    gps->push_back(GaussianProcess(inputs, observations.col(output_dim),
                                  std::move(cov_func)));
    (*gps)[output_dim].autoTune(conf);
  }
  return GP(std::move(gps), ga_conf);  
}

std::string GP::class_name() const
{
  return "GPTrainer";
}

void GP::to_xml(std::ostream &out) const
{
  autotune_conf.write("autotune_conf", out);
  ga_conf.write("ga_conf", out);
}

void GP::from_xml(TiXmlNode *node)
{
  autotune_conf.tryRead(node, "auto_tune_conf");
  ga_conf.tryRead(node, "ga_conf");
}

}
