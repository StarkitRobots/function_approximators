#pragma once

#include "rhoban_utils/serialization/json_serializable.h"

#include <Eigen/Core>

#include <memory>

namespace rhoban_fa
{

// Previous declaration
class FunctionApproximator;

/// Describe the interface of a FunctionApproximator trainer.
/// Can be serialized from/to xml files
/// Can be built with a TrainerFactory
class Trainer : public rhoban_utils::JsonSerializable
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

  /// Train the function approximator using initial_fa as a basis:
  /// - Default implementation is to ignore the initial_fa, but can be overriden
  virtual std::unique_ptr<FunctionApproximator>
  train(const Eigen::MatrixXd & inputs,
        const Eigen::MatrixXd & observations,
        const Eigen::MatrixXd & limits,
        const FunctionApproximator & initial_fa) const;

  /// Update the number of threads allowed for the trainer
  void setNbThreads(int nb_threads);

  virtual Json::Value toJson() const override;
  virtual void fromJson(const Json::Value & v, const std::string & dir_name) override;

protected:
  /// Throws an explicit logic_error if informations are not consistent
  void checkConsistency(const Eigen::MatrixXd & inputs,
                        const Eigen::MatrixXd & observations,
                        const Eigen::MatrixXd & limits) const;

  /// The number of threads allowed to the trainer
  int nb_threads;
};

}
