#include "starkit_fa/orthogonal_split.h"

#include "starkit_utils/io_tools.h"

namespace starkit_fa
{
OrthogonalSplit::OrthogonalSplit() : dim(-1), val(0)
{
}

OrthogonalSplit::OrthogonalSplit(int dim_, double val_) : dim(dim_), val(val_)
{
}

std::unique_ptr<Split> OrthogonalSplit::clone() const
{
  return std::unique_ptr<Split>(new OrthogonalSplit(dim, val));
}

int OrthogonalSplit::getNbElements() const
{
  return 2;
}

int OrthogonalSplit::getIndex(const Eigen::VectorXd& input) const
{
  // Checking for eventual issues
  if (dim < 0)
    throw std::logic_error("OrthogonalSplit::getIndex: negative split dim (uninitialized?)");
  if (input.rows() < dim)
  {
    std::ostringstream oss;
    oss << "OrthogonalSplit::getIndex(): input has not enough rows: " << input.rows() << " rows while split dim is "
        << dim;
    throw std::logic_error(oss.str());
  }
  // Computing index
  return input(dim) < val ? 0 : 1;
}

std::vector<Eigen::MatrixXd> OrthogonalSplit::splitSpace(const Eigen::MatrixXd& space) const
{
  std::vector<Eigen::MatrixXd> result(2);
  // Index 0: max is val
  result[0] = space;
  result[0](dim, 1) = val;
  // Index 1: min is val
  result[1] = space;
  result[1](dim, 0) = val;
  return result;
}

std::string OrthogonalSplit::toString() const
{
  std::ostringstream oss;
  oss << "(OrthogonalSplit: dim= " << dim << " | val = " << val << ")";
  return oss.str();
}

int OrthogonalSplit::getClassID() const
{
  return ID::Orthogonal;
}

int OrthogonalSplit::writeInternal(std::ostream& out) const
{
  int bytes_written = 0;
  bytes_written += starkit_utils::write<int>(out, dim);
  bytes_written += starkit_utils::write<double>(out, val);
  return bytes_written;
}

int OrthogonalSplit::read(std::istream& in)
{
  int bytes_read = 0;
  bytes_read += starkit_utils::read<int>(in, &dim);
  bytes_read += starkit_utils::read<double>(in, &val);
  return bytes_read;
}

}  // namespace starkit_fa
