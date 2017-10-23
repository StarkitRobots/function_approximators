#include "rosban_fa/linear_split.h"

#include "rosban_utils/io_tools.h"

//TODO: a check for coeffs value should be used to avoid the specific case where all elements are 0

namespace rosban_fa
{

LinearSplit::LinearSplit()
{
}

LinearSplit::LinearSplit(const Eigen::VectorXd & coeffs_, double offset_)
{
  double length =  coeffs_.norm();
  coeffs = coeffs_ / length;
  offset = offset_ / length;
}

LinearSplit::LinearSplit(const Eigen::VectorXd & coeffs_, const Eigen::VectorXd & point)
{
  coeffs = coeffs_.normalized();
  offset = -coeffs.dot(point);
}

std::unique_ptr<Split> LinearSplit::clone() const {
  return std::unique_ptr<Split>(new LinearSplit(coeffs, offset));
}

int LinearSplit::getNbElements() const
{
  return 2;
}

int LinearSplit::getIndex(const Eigen::VectorXd & input) const
{
  // Index is based on signed distance
  double dist = coeffs.dot(input) + offset;
  if (dist > 0) {
    return 1;
  }
  return 0;
}

std::vector<Eigen::MatrixXd> LinearSplit::splitSpace(const Eigen::MatrixXd & space) const
{
  (void) space;
  throw std::logic_error("LinearSplit::splitSpace: forbidden (cannot split into hyperrectangles)");
}

std::string LinearSplit::toString() const
{
  std::ostringstream oss;
  oss << "(LinearSplit: coeffs= " << coeffs.transpose() << ", offset= " << offset << ")";
  return oss.str();
}

int LinearSplit::getClassID() const
{
  return ID::Linear;
}

int LinearSplit::writeInternal(std::ostream & out) const
{
  int bytes_written = 0;
  int dim = coeffs.rows();
  bytes_written += rosban_utils::write<int>(out, dim);
  bytes_written += rosban_utils::writeDoubleArray(out, coeffs.data(), dim);
  bytes_written += rosban_utils::write<double>(out, offset);
  return bytes_written;
}

int LinearSplit::read(std::istream & in)
{
  int bytes_read = 0;
  int dim(0);
  bytes_read += rosban_utils::read<int>(in, &dim);
  coeffs = Eigen::VectorXd(dim);
  bytes_read += rosban_utils::readDoubleArray(in, coeffs.data(), dim);
  bytes_read += rosban_utils::read<double>(in, &offset);
  return bytes_read;
}

}
