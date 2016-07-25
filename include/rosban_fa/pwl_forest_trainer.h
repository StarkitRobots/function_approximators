#pragma once

#include "rosban_fa/trainer.h"

namespace rosban_fa
{

class PWLForestTrainer : public Trainer
{
  PWLForestTrainer();

  /// Update internal structure according to the provided samples
  virtual std::unique_ptr<FunctionApproximator>
  train(const Eigen::MatrixXd & inputs,
        const Eigen::MatrixXd & observations,
        const Eigen::MatrixXd & limits) const override;

  virtual std::string class_name() const override;
  virtual void to_xml(std::ostream &out) const override;
  virtual void from_xml(TiXmlNode *node) override;


private:
  int max_action_tiles;
};

}
