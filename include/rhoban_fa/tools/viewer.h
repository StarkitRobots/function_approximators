#include "rhoban_fa/function_approximator.h"

#include "rhoban_viewer/viewer.h"

#include <memory>

namespace rhoban_fa
{

class Viewer : public rhoban_viewer::Viewer {

public:
  /// fa_path should be readable by the FAFactory
  /// config_path should be a csv with n columns and 3 rows
  /// 1. headers (including value)
  /// 2. mins
  /// 3. maxs
  Viewer(const std::string& fa_path,
         const std::string& config_path,
         unsigned int width = 800, unsigned int height = 600);

  /// Update status message
  void updateStatus();
  virtual bool update() override;

  /// Number of dimensions of input
  int inputSize() const;

  /// Total number of parameters
  int nbParameters() const;

  /**
   * Jump to previous index if shift is pressed, next otherwise 
   */
  void navigate();

protected:

  /// Rescale rawValue from current limits to [0,1]
  double rescaleValue(double rawValue, int dim);

  /// Return the size of the step according to ratio used and currently selected dimension
  double getStep(double ratio);

  /// Update the limits and return true if limits have changed
  bool updateLimits(double step);

  /// Increase value of the selected item by ratio * (space_max - space_min)
  /// Note: In case this would result in min > max on the chosen dimension,
  ///       the value is set to the other bound value
  void increaseValue(double ratio);

  /// Set current value to space_max
  /// Note: If selected index is min, then current_min is set to current_max
  void valueToMax();

  /// Set current value to space_min
  /// Note: If selected index is max, then current_max is set to current_min
  void valueToMin();

  /// Switch a dimension status between free and locked
  void toggle();

  /// Update corners according to current limits
  void updateCorners();

  /// Build a tile with the provided arguments
  std::vector<Eigen::VectorXd> makeTile(const Eigen::MatrixXd & inputs,
                                        const Eigen::VectorXd & outputs,
                                        const std::vector<int> & samples_idx,
                                        const std::vector<int> & free_dims);

  void drawTiles();

  /// Return a list of the dimensions which are considered as 'free'
  std::vector<int> freeDimensions();

  /// Accessors to current limits
  const Eigen::MatrixXd & getCurrentLimits() const;

  /// Append dimension description to a stream
  void appendDim(int dim, std::ostream &out) const;

  /// Append dimension limits to a stream
  void appendLimits(int dim, std::ostream &out) const;

private:
  /// The function approximator which is currently displayed
  std::unique_ptr<FunctionApproximator> fa;

  /// The dimension on which the user is currently acting
  /// -1: no dimension selected
  ///  [0, dim_names -2[ : input
  /// dim_names - 3: output
  /// dim_names - 2: nb points by dim
  /// dim_names - 1: output_dimension (for several output)
  /// dim_names    : display mode [0:value, 1: stdDev]
  int dim_index;

  /// Internal index of the dimension:
  /// -1: Focus on the dimension itself
  ///  0: Focus on min
  ///  1: Focus on max
  int sub_dim_index;

  /// Which number of points is used per dimension
  int nb_points_by_dim;

  /// Which output dim is shown ?
  int output_dim;

  /// Dimension names
  std::vector<std::string> dim_names;

  /// Global limits for dimensions (including 'output', 'nb points by dim' and 'output dim')
  Eigen::MatrixXd space_limits;

  /// Current range of values for input/output
  Eigen::MatrixXd current_limits;

  /// For input, if dim is locked, it has a specific value
  /// For output, if dim is locked, it is considered as automatically chosen
  std::vector<bool> locked;

  /// Position of the corners of the tiles
  std::vector<std::vector<Eigen::VectorXd>> tiles;
  /// Color of corners
  std::vector<std::vector<rhoban_viewer::Color>> corners_color;
      
};
}
