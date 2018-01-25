#pragma once

#include "rosban_fa/function_approximator.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "tiny_dnn/tiny_dnn.h"
#pragma GCC diagnostic pop

namespace rosban_fa
{

class DNNApproximator : public FunctionApproximator {
public:
  typedef tiny_dnn::network<tiny_dnn::sequential> network;

  DNNApproximator();
  DNNApproximator(const network & nn, int input_dims, int output_dims, int nb_units);
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

  /// Update the structure of the neural network according to inner parameters
  static network buildNN(int input_dim, int output_dim, int nb_units);

private:
  /// The neural network used to predict approximation
  network nn;

  /// Size of input
  int input_dim;
  
  /// Size of output
  int output_dim;

  /// Nb elements in hidden layer
  int nb_units;
};

}
