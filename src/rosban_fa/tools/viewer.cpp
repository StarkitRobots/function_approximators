#include "rosban_fa/tools/viewer.h"

#include "rosban_fa/function_approximator_factory.h"

#include <rosban_utils/space_tools.h>
#include <rosban_utils/string_tools.h>

#include <fstream>
#include <iostream>

#include <SFML/OpenGL.hpp>

using rosban_viewer::Color;

namespace rosban_fa
{

Viewer::Viewer(const std::string& fa_path,
               const std::string& config_path,
               unsigned int width,
               unsigned int height)
  : rosban_viewer::Viewer(width, height),
    dim_index(-1),
    sub_dim_index(-1)
{
  std::cout << "Loading file '" << fa_path << "' as a function approximator" << std::endl;
  FunctionApproximatorFactory().loadFromFile(fa_path, fa);
  std::cout << "-> Loaded" << std::endl;

  // Read config file content
  std::ifstream configStream;
  configStream.open(config_path);
  std::vector<std::string> lines;
  while(configStream.good()) {
    std::string line;
    getline(configStream, line);
    lines.push_back(line);
  }
  configStream.close();
  if (lines.size() != 3) {
    throw std::runtime_error("Expecting 3 lines in config file: (header, mins, maxs)");
  }
  // Treat config file content
  std::vector<std::string> mins, maxs;
  dim_names = rosban_utils::split_string(lines[0], ',');
  mins      = rosban_utils::split_string(lines[1], ',');
  maxs      = rosban_utils::split_string(lines[2], ',');
  if (dim_names.size() != mins.size() || dim_names.size() != maxs.size()) {
    throw std::runtime_error("Inconsistent config file");
  }
  // Adding the 'points by dim' dimension
  dim_names.push_back("points by dim");
  mins.push_back("2");
  maxs.push_back("1000");
  // Adding the 'output dim' dimension
  dim_names.push_back("output dim");
  mins.push_back("0");
  maxs.push_back(std::to_string(fa->getOutputDim() - 1));
  // Adding the 'display type' dimension
  dim_names.push_back("display type");
  mins.push_back("0");
  maxs.push_back("1");
  // Setting space limits
  space_limits = Eigen::MatrixXd(dim_names.size(), 2);
  for (size_t dim = 0; dim < dim_names.size(); dim++) {
    space_limits(dim, 0) = std::stod(mins[dim]);
    space_limits(dim, 1) = std::stod(maxs[dim]);
  }

  // Initially, all input dimensions are locked on the middle value
  current_limits = Eigen::MatrixXd(nbParameters(), 2);
  locked = std::vector<bool>(nbParameters(), true);
  for (int dim = 0; dim < inputSize(); dim++) {
    double mid_val = (space_limits(dim, 0) + space_limits(dim, 1)) / 2;
    for (int sub_dim : {0,1}) {
      current_limits(dim, sub_dim) = mid_val;
    }
  }
  // Special behavior for output
  current_limits.row(inputSize()) = space_limits.row(inputSize());
  // Special behavior for 'nb points by dim'
  current_limits.row(inputSize() + 1) = Eigen::VectorXd::Constant(2, 100).transpose();
  // Special behavior for 'output_dim'
  current_limits.row(inputSize() + 2) = Eigen::VectorXd::Constant(2, 0).transpose();
  // Special behavior for 'display type'
  current_limits.row(inputSize() + 3) = Eigen::VectorXd::Constant(2, 0).transpose();

  // Initializing keyboard callbacks
  onKeyPress[sf::Keyboard::Tab].push_back([this](){ this->navigate(); });
  onKeyPress[sf::Keyboard::PageUp].push_back([this]()
                                             {
                                               this->increaseValue(0.01);
                                             });
  onKeyPress[sf::Keyboard::PageDown].push_back([this]()
                                               {
                                                 this->increaseValue(-0.01);
                                               });
  onKeyPress[sf::Keyboard::Home].push_back([this]()
                                           { this->valueToMax(); });
  onKeyPress[sf::Keyboard::End].push_back([this]()
                                          { this->valueToMin(); });
  onKeyPress[sf::Keyboard::T].push_back([this]()
                                        { this->toggle(); });

  // Default messages
  font.loadFromFile("monkey.ttf");//Follow it!
  status = sf::Text("Status", font);
  status.setCharacterSize(30);
  status.setColor(sf::Color::Red);

  // Update ground
  for(int i : {0,1}){
    groundLimits(i,0) = 0;
    groundLimits(i,1) = 1;
  }
}

int Viewer::inputSize() const
{
  return nbParameters() - 4;
}

int Viewer::nbParameters() const
{
  return dim_names.size();
}

double Viewer::rescaleValue(double rawValue, int dim)
{
  double min = current_limits(dim, 0);
  double max = current_limits(dim, 1);
  double delta = max - min;
  double rescaled = (rawValue - min) / delta;
  // Normalizing
  return std::min(1., std::max(rescaled, 0.));
}

double Viewer::getStep(double ratio)
{
  if (dim_index > inputSize()) {
    return ratio > 0 ? 1 : -1;
  }
  double min = space_limits(dim_index, 0);
  double max = space_limits(dim_index, 1);
  return (max - min) * ratio;
}

bool Viewer::updateLimits(double step)
{
  double space_min = space_limits(dim_index, 0);
  double space_max = space_limits(dim_index, 1);
  double old_min = current_limits(dim_index, 0);
  double old_max = current_limits(dim_index, 1);
  double new_min = old_min;
  double new_max = old_max;
  // Updating value when dimension is locked
  switch(sub_dim_index)
  {
    case -1:
      new_min = std::max(space_min, std::min(space_max, old_min + step));
      new_max = new_min;
      break;
    case 0:
      new_min = std::max(space_min, std::min(old_max, old_min + step));
      break;
    case 1:
      new_max = std::min(space_max, std::max(old_min, old_max + step));
      break;
    default:
      throw std::runtime_error("Invalid sub_dim_index");
  }
  current_limits(dim_index, 0) = new_min;
  current_limits(dim_index, 1) = new_max;
  return (new_min != old_min || new_max != old_max);
}

void Viewer::increaseValue(double ratio)
{
  // If no dimension is selected, do nothing
  if (dim_index == -1) return;
  // Applying eventual modifiers
  if (sf::Keyboard::isKeyPressed(sf::Keyboard::LShift)) { ratio *= 10; }
  if (sf::Keyboard::isKeyPressed(sf::Keyboard::LControl)) { ratio /= 10; }
  // Getting step size
  double step = getStep(ratio);
  // Updating value
  bool has_changed = updateLimits(step);
  // Update the structure to display if modification had an effect
  if (has_changed) updateCorners();
}

void Viewer::valueToMax()
{
  if (dim_index == -1) return;
  double max = space_limits(dim_index, 1);
  switch(sub_dim_index)
  {
    case -1:
      current_limits(dim_index, 0) = max;
      current_limits(dim_index, 1) = max;
      break;
    case 0:
      current_limits(dim_index, 0) = current_limits(dim_index,1);
      break;
    case 1:
      current_limits(dim_index, 1) = max;
      break;
    default:
      throw std::logic_error("Invalid value for sub_dim_index");
  }
  updateCorners();
}

void Viewer::valueToMin()
{
  if (dim_index == -1) return;
  double min = space_limits(dim_index, 0);
  switch(sub_dim_index)
  {
    case -1:
      current_limits(dim_index, 0) = min;
      current_limits(dim_index, 1) = min;
      break;
    case 0:
      current_limits(dim_index, 0) = min;
      break;
    case 1:
      current_limits(dim_index, 1) = current_limits(dim_index, 0);
      break;
    default:
      throw std::logic_error("Invalid value for sub_dim_index");
  }
  updateCorners();
}

void Viewer::toggle()
{
  // If no dim selected, do nothing
  if (dim_index == -1) return;
  // 'nb points by dim' and 'output dim' are not affected by toggle
  if (dim_index > inputSize()) return;
  // toggle locked status
  locked[dim_index] = !locked[dim_index];
  // If locked, set min and max to the middle value
  if (locked[dim_index])
  {
    double mid_val = (space_limits(dim_index, 0) + space_limits(dim_index, 1)) / 2;
    for (int sub_dim : {0,1})
    {
      current_limits(dim_index, sub_dim) = mid_val;
    }
    // Min and max cannot be selected since dim is locked
    sub_dim_index = -1;
  }
  // If unlocked, set min and max to extremum values
  else
  {
    for (int sub_dim : {0,1})
    {
      current_limits(dim_index, sub_dim) = space_limits(dim_index, sub_dim);
    }
  }
  updateCorners();
}

std::vector<int> Viewer::freeDimensions()
{
  std::vector<int> result;
  for (int d = 0; d < inputSize(); d++) {
    if (!locked[d]) {
      result.push_back(d);
    }
  }
  return result;
}

const Eigen::MatrixXd & Viewer::getCurrentLimits() const
{
  return current_limits;
}

void Viewer::updateCorners()
{
  tiles.clear();
  corners_color.clear();

  // Local limits
  Eigen::MatrixXd input_limits = getCurrentLimits().block(0,0,inputSize(),2);

  std::vector<int> freeDims = freeDimensions();

  if (freeDims.size() > 2 || freeDims.size() < 1) { return;}

  int samples_by_free_dim = current_limits(inputSize() + 1, 0);
  int output_dim = current_limits(inputSize() + 2, 0);
  // All dimensions have a single point, except free dimensions
  std::vector<int> samples_by_dim(inputSize(), 1);
  for (int dim : freeDims) {
    samples_by_dim[dim] = samples_by_free_dim;
  }

  std::cout << "Discretizing the state space inside limits" << std::endl;
  Eigen::MatrixXd inputs;
  inputs = rosban_utils::discretizeSpace(input_limits, samples_by_dim);
  std::cout << "-> #Inputs: " << inputs.cols() << std::endl;
  std::cout << "Computing outputs" << std::endl;
  Eigen::VectorXd outputs(inputs.cols());
  double min_output = std::numeric_limits<double>::max();
  double max_output = std::numeric_limits<double>::lowest();
  for (int point = 0; point < inputs.cols(); point++) {
    double output, mean, var;
    fa->predict(inputs.col(point), output_dim, mean, var);
    int display_type = current_limits(inputSize() + 3, 0);
    switch(display_type){
      case 0:
        output = mean;
        break;
      case 1:
        output = std::sqrt(var);
        break;
      default:
        throw std::logic_error("Invalid value for display type");
    }
    outputs(point) = output;
    min_output = std::min(output, min_output);
    max_output = std::max(output, max_output);
  }
  // Auto mode, update output limits for value according to content
  if (locked[inputSize()])
  {
    current_limits(inputSize(), 0) = min_output;//std::max(min_output, space_limits(inputSize(),0));
    current_limits(inputSize(), 1) = max_output;//std::min(max_output, space_limits(inputSize(),1));
  }

  // Creating Tiles
  std::cout << "Creating tiles" << std::endl;
  for (int dim1_idx = 0; dim1_idx < samples_by_free_dim - 1; dim1_idx++) {
    // Case where there is only 1 dimension
    if (freeDims.size() == 1) {
      std::vector<int> samples_idx = { dim1_idx, dim1_idx + 1};
      tiles.push_back(makeTile(inputs, outputs, samples_idx, freeDims));
      continue;
    }
    for (int dim2_idx = 0; dim2_idx < samples_by_free_dim - 1; dim2_idx++) {
      std::vector<int> samples_idx =
        {
          dim1_idx * samples_by_free_dim + dim2_idx,
          (dim1_idx + 1) * samples_by_free_dim + dim2_idx,
          (dim1_idx + 1) * samples_by_free_dim + dim2_idx +1,
          dim1_idx * samples_by_free_dim + dim2_idx + 1
        };
      tiles.push_back(makeTile(inputs, outputs, samples_idx, freeDims));
    }
  }
  std::cout << "#Tiles: " << tiles.size() << std::endl;

  // Adding color color
  for (size_t tileId = 0; tileId < tiles.size(); tileId++) {
    std::vector<Eigen::VectorXd> & projectedTile = tiles[tileId];
    std::vector<Color> tileCornersColor;
    for (size_t cornerId = 0; cornerId < projectedTile.size(); cornerId++) {
      double output = projectedTile[cornerId](2);
      Color color;
      double altColor = 1.0 - 2 * std::fabs(output - 0.5);
      if (output > 0.5) {
        color = Color(1, altColor, altColor);
      }
      else {
        color = Color(altColor, altColor, 1);
      }
      tileCornersColor.push_back(color);
    }
    corners_color.push_back(tileCornersColor);
  }
}

std::vector<Eigen::VectorXd> Viewer::makeTile(const Eigen::MatrixXd & inputs,
                                              const Eigen::VectorXd & outputs,
                                              const std::vector<int> & samples_idx,
                                              const std::vector<int> & free_dims)
{
  if (free_dims.size() != 1 && free_dims.size() != 2) {
    throw std::runtime_error("Invalid number of free_dims in makeTile");
  }
  if (free_dims.size() == 1) {
    if (samples_idx.size() != 2) {
      throw std::runtime_error("Invalid number of samples in makeTile");
    }
    int dim = free_dims[0];
    double x1 = rescaleValue(inputs(dim, samples_idx[0]), dim);
    double x2 = rescaleValue(inputs(dim, samples_idx[1]), dim);
    double val1 = rescaleValue(outputs(samples_idx[0]), inputSize());
    double val2 = rescaleValue(outputs(samples_idx[1]), inputSize());
    std::vector<Eigen::VectorXd> tile;
    tile.push_back(Eigen::Vector3d(x1, 0, val1));
    tile.push_back(Eigen::Vector3d(x2, 0, val2));
    tile.push_back(Eigen::Vector3d(x2, 1, val2));
    tile.push_back(Eigen::Vector3d(x1, 1, val1));
    return tile;
  }
  // 2 free dims
  if (samples_idx.size() != 4) {
    throw std::runtime_error("Invalid number of samples in makeTile");
  }
  int dim1 = free_dims[0];
  int dim2 = free_dims[1];
  std::vector<Eigen::VectorXd> tile;
  for (int idx : samples_idx) {
    double x = rescaleValue(inputs(dim1, idx), dim1);
    double y = rescaleValue(inputs(dim2, idx), dim2);
    double val = rescaleValue(outputs(idx), inputSize());
    tile.push_back(Eigen::Vector3d(x, y, val));
  }
  return tile;
}
void Viewer::navigate()
{
  //Shift-tab not handled in sfml 2.0
  if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::LControl)) {
    // If we can jump to previous sub_dim, it is enough
    if (sub_dim_index > -1)
    {
      sub_dim_index--;
    }
    // Jump to previous dimension
    else
    {
      dim_index--;
      // Circular list
      if (dim_index == -2) {
        dim_index = nbParameters() - 1;
      }
      // If final dimension is not locked, use last sub_dim_index
      if (dim_index >= 0 && !locked[dim_index])
        sub_dim_index = 1;
      else
        sub_dim_index = -1;
    }
  }
  // Forward case
  else {
    // If we can jump to next sub_dim, it is enough
    if (dim_index >= 0 && !locked[dim_index] && sub_dim_index < 1)
    {
      sub_dim_index++;
    }
    // Jump to next dim
    else
    {
      dim_index++;
      // circular list
      if (dim_index >= nbParameters()) {
        dim_index = -1;
      }
      sub_dim_index = -1;
    }
  }
}

void Viewer::drawTiles()
{
  glBegin(GL_QUADS);
  for (size_t tileId = 0; tileId < tiles.size(); tileId++) {
    const std::vector<Eigen::VectorXd>& points = tiles[tileId];
    const std::vector<Color>& colors = corners_color[tileId];
    for (size_t pId : {0,1,2,3}) {//Ordering matters for opengl
      const Eigen::VectorXd& p = points[pId];
      const Color& c = colors[pId];
      glColor3f(c.r, c.g, c.b);
      glVertex3f(p(0), p(1), p(2));
    }
  }
  glEnd();
}

void Viewer::appendLimits(int dim, std::ostream &out) const
{
  std::vector<std::pair<std::string,int>> names = {{"min", 0}, {"max", 1}};
  for (const auto & entry : names)
  {
    // Print selection indicator
    if (dim == dim_index && sub_dim_index == entry.second)
      out << "->    ";
    else
      out << "      ";
    // Print limits itself
    out << entry.first << ": " << current_limits(dim, entry.second) << std::endl;
  }     
}

void Viewer::appendDim(int dim, std::ostream &out) const
{
  bool input = dim != inputSize();
  // Printing selection indicator
  if (dim_index == dim && sub_dim_index == -1)
    out << "->";
  else
    out << "  ";
  // Printing name and limits
  out << dim_names[dim] << ": ";//Padding would be nice
  out << " [" << space_limits(dim,0) << "," << space_limits(dim,1) << "] ";
  // Status dependant message
  if (locked[dim])
  {
    // Just print information on lock value
    if (input)
    {
      out << "Locked at " << current_limits(dim,0) << ": " << std::endl;
    }
    // Print used bounds for output
    else
    {
      out << "Auto" << std::endl;
      appendLimits(dim, out);
    }
  }
  // Dimension is free
  else
  {
    // Just print information on lock value
    out << "Manual" << std::endl;
    appendLimits(dim, out);
  }
}

void Viewer::updateStatus()
{
  std::ostringstream oss;
  // Input:
  for (int dim = 0; dim < nbParameters(); dim++)
  {
    appendDim(dim, oss);
  }
  rosban_viewer::Viewer::updateStatus(oss.str());
}

bool Viewer::update()
{
  updateStatus();
  drawTiles();
  return rosban_viewer::Viewer::update();
}

}
