#include "rosban_fa/split.h"

namespace rosban_fa
{

std::vector<Eigen::MatrixXd> Split::splitEntries(const Eigen::MatrixXd & input) const
{
  // Main variables
  int nb_separations = getNbElements();
  int input_dim = input.rows();
  // First pass to extract indices
  std::vector<std::vector<int>> indices(nb_separations);
  for (int col = 0; col < input.cols(); col++)
  {
    indices[getIndex(input.col(col))].push_back(col);
  }
  // Second pass to emplace columns
  std::vector<Eigen::MatrixXd> result(nb_separations);
  for (int sep_idx = 0; sep_idx < nb_separations; sep_idx++)
  {
    result[sep_idx] = Eigen::MatrixXd(input_dim, indices[sep_idx].size());
    for (size_t i = 0; i < indices[sep_idx].size(); i++)
    {
      result[sep_idx].col(i) = input.col(indices[sep_idx][i]);
    }
  }
  return result;
}

}
