#pragma once

#include "rosban_fa/trainer.h"

#include "rosban_gp/gradient_ascent/randomized_rprop.h"

namespace rosban_fa
{

class GPForestTrainer : public Trainer
{
public:
  /// Which type of GPForest is used
  /// - SQRT: |samples|^{1/2} samples per node min
  /// - CURT: |samples|^{1/3} samples per node min
  /// - LOG2: log_2(|samples|)  sampels per node min
  enum class Type
  { SQRT, CURT, LOG2};

  GPForestTrainer(Type t = Type::LOG2);

  virtual std::unique_ptr<FunctionApproximator>
  train(const Eigen::MatrixXd & inputs,
        const Eigen::MatrixXd & observations,
        const Eigen::MatrixXd & limits) const override;

  virtual std::string class_name() const override;
  virtual void to_xml(std::ostream &out) const override;
  virtual void from_xml(TiXmlNode *node) override;

private:
  Type type;
  int nb_threads;

  rosban_gp::RandomizedRProp::Config autotune_conf;
  rosban_gp::RandomizedRProp::Config ga_conf;

  GPForestTrainer::Type loadType(const std::string &s);
};
std::string to_string(GPForestTrainer::Type type);


}
