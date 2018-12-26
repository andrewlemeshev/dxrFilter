#ifndef CAMERA_H
#define CAMERA_H

#include "Shared.h"

#define DEFAULT_CAMERA_SPEED 7.0f

#define MCS_TO_SEC(dt) float(dt) / 1000000.0f

class Camera {
public:
  Camera(const glm::vec4 &pos);
  ~Camera();

  void setUpVec(const glm::vec4 &worldUp);

  // что тут?
  void handleMouseInput(const float &horizontalAngle, const float &verticalAngle);

  void forward(const uint64_t &time, const float &speed = DEFAULT_CAMERA_SPEED);
  void backward(const uint64_t &time, const float &speed = DEFAULT_CAMERA_SPEED);
  void stepright(const uint64_t &time, const float &speed = DEFAULT_CAMERA_SPEED);
  void stepleft(const uint64_t &time, const float &speed = DEFAULT_CAMERA_SPEED);
  void upward(const uint64_t &time, const float &speed = DEFAULT_CAMERA_SPEED);
  void downward(const uint64_t &time, const float &speed = DEFAULT_CAMERA_SPEED);

  void calcView();

  glm::vec4 position();
  glm::vec4 direction();
  glm::mat4 view();
private:
  glm::vec4 pos;

  glm::vec4 front;
  glm::vec4 right;
  glm::vec4 up;

  glm::vec4 worldUp;

  glm::mat4 viewMat;
};

#endif