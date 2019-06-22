#pragma once

#include "starkit_fa/trainer.h"

#include "starkit_gp/gradient_ascent/randomized_rprop.h"

namespace starkit_fa
{
class GPTrainer : public Trainer
{
public:
  GPTrainer();

  virtual std::unique_ptr<FunctionApproximator> train(const Eigen::MatrixXd& inputs,
                                                      const Eigen::MatrixXd& observations,
                                                      const Eigen::MatrixXd& limits) const override;

  virtual std::string getClassName() const override;
  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value& v, const std::string& dir_name) override;

private:
  starkit_gp::RandomizedRProp::Config autotune_conf;
  starkit_gp::RandomizedRProp::Config ga_conf;
};

}  // namespace starkit_fa
