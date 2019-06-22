#pragma once

#include "starkit_utils/serialization/stream_serializable.h"

#include <Eigen/Core>

#include <memory>
#include <vector>

namespace starkit_fa
{
/// This interface do not require to cut the space on which the split is applied
/// because it is not mandatory for splits (e.g. orthogonal split only require
/// to store the dimension id and the value to be valid)
class Split : public starkit_utils::StreamSerializable
{
public:
  enum ID : int
  {
    Orthogonal = 1,
    Point = 2,
    Fake = 3,
    Linear = 4
  };

  virtual std::unique_ptr<Split> clone() const = 0;

  /// Return the total number of elements of the chosen split
  virtual int getNbElements() const = 0;

  /// Return the index of the subspace containing to the given input
  virtual int getIndex(const Eigen::VectorXd& input) const = 0;

  /// Separate all elements of input
  /// return a vector v in which v[i] is the set of elements which
  std::vector<Eigen::MatrixXd> splitEntries(const Eigen::MatrixXd& input) const;

  /// Return the hyperrectangles corresponding to the subspaces of the given
  /// space as defined by the getIndex function
  virtual std::vector<Eigen::MatrixXd> splitSpace(const Eigen::MatrixXd& space) const = 0;

  virtual std::string toString() const = 0;
};

}  // namespace starkit_fa
