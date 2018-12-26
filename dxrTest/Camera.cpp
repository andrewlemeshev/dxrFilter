#include "Camera.h"

#include <cmath>

Camera::Camera(const glm::vec4 &pos) {
  this->pos = pos;
  this->front = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
  this->worldUp = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
}

Camera::~Camera() {}

void Camera::setUpVec(const glm::vec4 &worldUp) {
  this->worldUp = worldUp;
}

// что тут?
void Camera::handleMouseInput(const float &horizontalAngle, const float &verticalAngle) {
  float hAngle = horizontalAngle;
  float vAngle = verticalAngle;
  if (hAngle > glm::two_pi<float>()) hAngle -= glm::two_pi<float>();
  if (hAngle < 0.0f)                 hAngle += glm::two_pi<float>();
  
  //if (vAngle > glm::half_pi<float>() - EPSILON) 
  vAngle = glm::min(vAngle,  glm::half_pi<float>() - EPSILON);
  vAngle = glm::max(vAngle, -glm::half_pi<float>() + EPSILON);
  
  front = glm::normalize(glm::vec4(
    -glm::cos(vAngle) * glm::sin(hAngle),
    -glm::sin(vAngle),
    glm::cos(vAngle) * glm::cos(hAngle),
    0.0f
  ));

  right = glm::normalize(glm::vec4(glm::cross(glm::vec3(front), glm::vec3(worldUp)), 0.0f));
  up = glm::normalize(glm::vec4(glm::cross(glm::vec3(front), glm::vec3(right)), 0.0f));
}

void Camera::forward(const uint64_t &time, const float &speed) {
  const float dt = MCS_TO_SEC(time);

  pos += front * dt * speed;
}

void Camera::backward(const uint64_t &time, const float &speed) {
  const float dt = MCS_TO_SEC(time);

  pos -= front * dt * speed;
}

void Camera::stepright(const uint64_t &time, const float &speed) {
  const float dt = MCS_TO_SEC(time);

  pos -= right * dt * speed;
}

void Camera::stepleft(const uint64_t &time, const float &speed) {
  const float dt = MCS_TO_SEC(time);

  pos += right * dt * speed;
}

void Camera::upward(const uint64_t &time, const float &speed) {
  const float dt = MCS_TO_SEC(time);

  pos += up * dt * speed;
}

void Camera::downward(const uint64_t &time, const float &speed) {
  const float dt = MCS_TO_SEC(time);

  pos -= up * dt * speed;
}

void Camera::calcView() {
  viewMat = glm::lookAt(glm::vec3(pos), glm::vec3(pos + front), glm::vec3(worldUp));
}

glm::vec4 Camera::position() {
  return pos;
}

glm::vec4 Camera::direction() {
  return front;
}

glm::mat4 Camera::view() {
  return viewMat;
}