#pragma once

#include "rosban_fa/split.h"

namespace rosban_fa
{

/// A point split is a split along all dimensions, following this idea:
/// index = Sum((input(dim) > point(dim)) * 2^dim):
class PointSplit : public Split
{
public:

  PointSplit();
  PointSplit(const Eigen::VectorXd & point);

  virtual int getNbElements() const override;

  virtual int getIndex(const Eigen::VectorXd & input) const override;

  virtual std::vector<Eigen::MatrixXd> splitSpace(const Eigen::MatrixXd & space) const;

  // Stream Serialization
  virtual int getClassID() const override;
  virtual int writeInternal(std::ostream & out) const override;
  virtual int read(std::istream & in) override;

private:
  Eigen::VectorXd split_point;
};

}
