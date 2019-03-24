#include "rhoban_fa/trainer.h"

#include "rhoban_fa/function_approximator.h"

#include <sstream>
#include <stdexcept>

namespace rhoban_fa
{
Trainer::Trainer() : nb_threads(1)
{
}

Trainer::~Trainer()
{
}

std::unique_ptr<FunctionApproximator> Trainer::train(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& observations,
                                                     const Eigen::MatrixXd& limits,
                                                     const FunctionApproximator& initial_fa) const
{
  (void)initial_fa;
  return train(inputs, observations, limits);
}

void Trainer::setNbThreads(int new_nb_threads)
{
  nb_threads = new_nb_threads;
}

Json::Value Trainer::toJson() const
{
  Json::Value v;
  v["nb_threads"] = nb_threads;
  return v;
}

void Trainer::fromJson(const Json::Value& v, const std::string& dir_name)
{
  (void)dir_name;
  rhoban_utils::tryRead(v, "nb_threads", &nb_threads);
}

void Trainer::checkConsistency(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& observations,
                               const Eigen::MatrixXd& limits) const
{
  if (inputs.cols() != observations.rows())
  {
    std::ostringstream oss;
    oss << "FunctionApproximator::checkConsistency: inconsistent number of samples: "
        << "inputs.cols() != observations.rows() "
        << "(" << inputs.cols() << " != " << observations.rows() << ")";
    throw std::logic_error(oss.str());
  }
  if (limits.rows() != inputs.rows())
  {
    std::ostringstream oss;
    oss << "FunctionApproximator::checkConsistency: inconsistent dimension for input: "
        << "inputs.rows() != limits.rows() "
        << "(" << inputs.rows() << " != " << limits.rows() << ")";
    throw std::logic_error(oss.str());
  }
  if (limits.cols() != 2)
  {
    std::ostringstream oss;
    oss << "FunctionApproximator::checkConsistency: invalid dimensions for limits: "
        << "expecting 2 columns and received " << limits.rows();
    throw std::logic_error(oss.str());
  }
}

}  // namespace rhoban_fa
