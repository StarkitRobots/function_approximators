#pragma once

#include "rosban_fa/function_approximator.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#include "tiny_dnn/tiny_dnn.h"
#pragma GCC diagnostic pop

namespace rosban_fa
{

/// A function approximator based on multi-layer perceptron
///
/// While it is based on tiny_dnn for training, it uses its own computation to
/// predict the value because prediction in tiny_dnn is not 'const'
class DNNApproximator : public FunctionApproximator {
public:
  typedef tiny_dnn::network<tiny_dnn::sequential> network;

  DNNApproximator();
  DNNApproximator(const network & nn, int input_dims, int output_dims,
                  const std::vector<int> layer_units);
  DNNApproximator(const DNNApproximator & other);

  virtual std::unique_ptr<FunctionApproximator> clone() const;

  int getOutputDim() const override;

  virtual void predict(const Eigen::VectorXd & input,
                       Eigen::VectorXd & mean,
                       Eigen::MatrixXd & covar) const override;

  virtual void gradient(const Eigen::VectorXd & input,
                        Eigen::VectorXd & gradient) const override;

  virtual void getMaximum(const Eigen::MatrixXd & limits,
                          Eigen::VectorXd & input,
                          double & output) const override;

  virtual int getClassID() const override;
  virtual int writeInternal(std::ostream & out) const override;
  virtual int read(std::istream & in) override;

  /// Structure is as following:
  /// input_dim -> layer_units[0]
  /// ...
  /// layer_units[k] -> output_dim
  static network buildNN(int input_dim, int output_dim,
                         const std::vector<int> & layer_units);

private:
  /// Synchronize hidden_layer_weights and final_layer_weights with current nn
  void updateWeightsFromNN();


  /// The neural network used to predict approximation
  network nn;

  /// Size of input
  int input_dim;
  
  /// Size of output
  int output_dim;

  /// Nb elements in hidden layer
  std::vector<int> layer_units;

  /// Weights of each layer of the DNN (coeffs, bias)
  std::vector<std::pair<Eigen::MatrixXd,Eigen::VectorXd>> layer_weights;
};

}
