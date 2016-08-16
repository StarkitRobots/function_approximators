#pragma once

#include "rosban_utils/serializable.h"

#include <Eigen/Core>

#include <memory>

namespace rosban_fa
{

// Previous declaration
class FunctionApproximator;

/// Describe the interface of a FunctionApproximator trainer.
/// Can be serialized from/to xml files
/// Can be built with a TrainerFactory
class Trainer : public rosban_utils::Serializable
{
public:

  Trainer();
  virtual ~Trainer();

  /// Train the function approximator with the provided set of N samples
  /// inputs: a I by N matrix, each column is a different input
  /// outputs: a N by O matrix, each row is a different output 
  /// limits:
  /// - column 0 is min
  /// - column 1 is max
  virtual std::unique_ptr<FunctionApproximator>
  train(const Eigen::MatrixXd & inputs,
        const Eigen::MatrixXd & observations,
        const Eigen::MatrixXd & limits) const = 0;

  /// Update the number of threads allowed for the trainer
  void setNbThreads(int nb_threads);

  virtual void to_xml(std::ostream &out) const override;
  virtual void from_xml(TiXmlNode *node) override;

protected:
  /// Throws an explicit logic_error if informations are not consistent
  void checkConsistency(const Eigen::MatrixXd & inputs,
                        const Eigen::MatrixXd & observations,
                        const Eigen::MatrixXd & limits) const;

  /// The number of threads allowed to the trainer
  int nb_threads;
};

}
