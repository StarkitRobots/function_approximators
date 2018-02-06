#pragma once

#include "rhoban_fa/split.h"

namespace rhoban_fa
{

/// For orthogonal splits, only one dimension is considered and values
/// if  input(dim) < val:
///   index = 0
/// else
///   index = 1
class OrthogonalSplit : public Split
{
public:

  OrthogonalSplit();
  OrthogonalSplit(int dim, double val);

  virtual std::unique_ptr<Split> clone() const override;

  virtual int getNbElements() const override;

  virtual int getIndex(const Eigen::VectorXd & input) const override;

  virtual std::vector<Eigen::MatrixXd> splitSpace(const Eigen::MatrixXd & space) const;

  // Stream Serialization
  virtual int getClassID() const override;
  virtual int writeInternal(std::ostream & out) const override;
  virtual int read(std::istream & in) override;

  virtual std::string toString() const override;

  int dim;
  double val;
};

}
