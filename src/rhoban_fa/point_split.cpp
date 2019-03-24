#include "rhoban_fa/point_split.h"

#include "rhoban_utils/io_tools.h"

namespace rhoban_fa
{
PointSplit::PointSplit()
{
}

PointSplit::PointSplit(const Eigen::VectorXd& point) : split_point(point)
{
}

std::unique_ptr<Split> PointSplit::clone() const
{
  return std::unique_ptr<Split>(new PointSplit(split_point));
}

int PointSplit::getNbElements() const
{
  return std::pow(2, split_point.rows());
}

int PointSplit::getIndex(const Eigen::VectorXd& input) const
{
  // Checking for eventual issues
  if (split_point.rows() <= 0)
    throw std::logic_error("PointSplit::getIndex: split point has no rows");
  // Checking for mismatch
  if (input.rows() != split_point.rows())
  {
    std::ostringstream oss;
    oss << "PointSplit::getIndex(): input has not enough rows: " << input.rows() << " rows while split_point has "
        << split_point.rows() << "rows";
    throw std::logic_error(oss.str());
  }
  // computing index
  int index = 0;
  int gain = 1;
  for (int dim = 0; dim < split_point.rows(); dim++)
  {
    if (input(dim) > split_point(dim))
    {
      index += gain;
    }
    gain *= 2;
  }
  return index;
}

std::vector<Eigen::MatrixXd> PointSplit::splitSpace(const Eigen::MatrixXd& space) const
{
  if (space.rows() != split_point.rows())
  {
    std::ostringstream oss;
    oss << "PointSplit::splitSpace: space dim (" << space.rows() << ") "
        << "does not match split dim (" << split_point.rows() << ")";
    throw std::logic_error(oss.str());
  }
  std::vector<Eigen::MatrixXd> result(std::pow(2, split_point.rows()));
  for (size_t index = 0; index < result.size(); index++)
  {
    result[index] = space;
    int ratio = 1;
    for (int dim = 0; dim < split_point.rows(); dim++)
    {
      // max side
      if ((index / ratio) % 2 == 1)
      {
        result[index](dim, 0) = split_point(dim);
      }
      // min side
      else
      {
        result[index](dim, 1) = split_point(dim);
      }
      ratio *= 2;
    }
  }
  return result;
}

std::string PointSplit::toString() const
{
  std::ostringstream oss;
  oss << "(PointSplit: point= " << split_point.transpose() << ")";
  return oss.str();
}

int PointSplit::getClassID() const
{
  return ID::Point;
}

int PointSplit::writeInternal(std::ostream& out) const
{
  int bytes_written = 0;
  int dim = split_point.rows();
  bytes_written += rhoban_utils::write<int>(out, dim);
  bytes_written += rhoban_utils::writeDoubleArray(out, split_point.data(), dim);
  return bytes_written;
}

int PointSplit::read(std::istream& in)
{
  int bytes_read = 0;
  int dim(0);
  bytes_read += rhoban_utils::read<int>(in, &dim);
  split_point = Eigen::VectorXd(dim);
  bytes_read += rhoban_utils::readDoubleArray(in, split_point.data(), dim);
  return bytes_read;
}

}  // namespace rhoban_fa
