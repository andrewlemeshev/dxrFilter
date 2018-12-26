// dxrTest.cpp : Defines the entry point for the application.
//

#include "stdafx.h"
//#include "dxrTest.h"

#include "Window.h"
#include "Render.h"
#include "Buffer.h"
#include "Camera.h"

#include <random>
#include <unordered_map>
#include <functional>
#include <chrono>

const uint32_t width = 1280;
const uint32_t height = 720;
const bool fullScreen = false;

const uint32_t boxCount = 1;
const uint32_t icosahedronCount = 1;
const uint32_t coneCount = 1;

// сюда на вход должны приходить по идее буферы которые мы заполняем
void initScene(GPUBuffer<ComputeData> &boxBuffer, GPUBuffer<ComputeData> &icosahedronBuffer, GPUBuffer<ComputeData> &coneBuffer);

// сюда передаем рендер и буферы (достаточно?) 
void mainLoop(const Window &window, DX12Render &render, GPUBuffer<ComputeData> &boxBuffer, GPUBuffer<ComputeData> &icosahedronBuffer, GPUBuffer<ComputeData> &coneBuffer);

void setDefaultKeyMap(Camera &camera, std::unordered_map<int, std::function<void(const uint64_t &)>> &keyMap);
void updateMouseMovement(const uint64_t &time, const Window &window, Camera &camera);
void updateKeyPressing(const uint64_t &time, std::unordered_map<int, std::function<void(const uint64_t &)>> &keyMap);

//int APIENTRY wWinMain(HINSTANCE hInstance, HINSTANCE, LPWSTR, int nCmdShow) {
int __stdcall wWinMain(HINSTANCE hInstance, HINSTANCE, LPWSTR, int nCmdShow) {
  Window window;
  window.init(hInstance, nCmdShow, width, height, fullScreen);

  DX12Render render;
  render.init(window.getHWND(), window.getWidth(), window.getHeight(), window.isFullscreen());
  render.recreatePSO();

  GPUBuffer<ComputeData> boxBuffer;
  boxBuffer.construct(render.getDevice(), boxCount);
  GPUBuffer<ComputeData> icosahedronBuffer;
  icosahedronBuffer.construct(render.getDevice(), icosahedronCount);
  GPUBuffer<ComputeData> coneBuffer;
  coneBuffer.construct(render.getDevice(), coneCount);

  // здесь мы инитим объекты (ну то есть создаем какое-то количество боксов, конусов и сфер)
  // в принципе это 3 массива (причем наверное одинаковых (?))
  initScene(boxBuffer, icosahedronBuffer, coneBuffer);

  const uint32_t instanceCount = boxBuffer.size() + icosahedronBuffer.size() + coneBuffer.size();
  render.prepareRender(instanceCount, glm::mat4(1.0f));
  render.computePartHost(boxBuffer, icosahedronBuffer, coneBuffer);

  render.initRT(window.getWidth(), window.getHeight(), boxBuffer, icosahedronBuffer, coneBuffer);
  render.initFilter(window.getWidth(), window.getHeight());

  mainLoop(window, render, boxBuffer, icosahedronBuffer, coneBuffer);

  return 0;
}

//LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
//  switch (message) {
//    case WM_COMMAND: {
//      int wmId = LOWORD(wParam);
//      // Parse the menu selections:
//      switch (wmId) {
//        case IDM_ABOUT:
//        DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
//        break;
//        case IDM_EXIT:
//        DestroyWindow(hWnd);
//        break;
//        default:
//        return DefWindowProc(hWnd, message, wParam, lParam);
//      }
//    }
//    break;
//    case WM_PAINT: {
//      PAINTSTRUCT ps;
//      HDC hdc = BeginPaint(hWnd, &ps);
//      // TODO: Add any drawing code that uses hdc here...
//      EndPaint(hWnd, &ps);
//    }
//    break;
//    case WM_DESTROY:
//      PostQuitMessage(0);
//    break;
//    default:
//    return DefWindowProc(hWnd, message, wParam, lParam);
//  }
//  return 0;
//}

enum DIR : uint32_t {
  RIGHT = 0,
  FORWARD,
  LEFT,
  BACKWARD,
  DIR_COUNT
};

void mainLoop(const Window &window, DX12Render &render, GPUBuffer<ComputeData> &boxBuffer, GPUBuffer<ComputeData> &icosahedronBuffer, GPUBuffer<ComputeData> &coneBuffer) {
  MSG msg;
  ZeroMemory(&msg, sizeof(MSG));

  static glm::vec2 oldMousePos;

  //while (true) {
  //  if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE | PM_NOYIELD)) {
  //    bool handled = false;
  //    // короч каждую клавишу нужно обрабатывать отдельно
  //    if (msg.message >= WM_MOUSEFIRST && msg.message <= WM_MOUSELAST) handled = MyHandleMouseEvent(&msg);
  //    else if (msg.message >= WM_KEYFIRST && msg.message <= WM_KEYLAST) handled = MyHandleKeyEvent(&msg);
  //    else if (msg.message == WM_QUIT) break;

  //    // короче это какой то триндец лучше наверное все же glfw использовать
  //    // лучше glfw все же
  //    // там тоже с кондачка не сделаешь, нужно повтыкать

  //    // че с мышкой? есть WM_MOUSEMOVE и видимо нужно под ним все делать
  //    POINT p;
  //    if (GetCursorPos(&p)) {
  //      //cursor position now in p.x and p.y
  //    }

  //    ScreenToClient(window.getHWND(), &p);



  //    if (!handled) {
  //      TranslateMessage(&msg);
  //      DispatchMessage(&msg);
  //    }
  //  } else {
  //    
  //  }
  //}

  const float speed = 0.05f;
  const float xMax = 10.0f;
  const float zMax = 10.0f;
  const float distance = 20.0f;

  glm::vec3 pos = glm::vec3(1.0f, 3.0f, -10.0f);
  glm::vec3 move = glm::vec3(speed, 0.0f, 0.0f);
  DIR dirEnum = RIGHT;

  Camera camera(glm::vec4(pos, 1.0f));
  std::unordered_map<int, std::function<void(const uint64_t &)>> keyMap;
  setDefaultKeyMap(camera, keyMap);

  glfwSetInputMode(window.getWindow(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);

  //std::chrono::time_point<std::chrono::steady_clock> start;
  auto start = std::chrono::steady_clock::now();
  auto end = std::chrono::steady_clock::now();

  while (!glfwWindowShouldClose(window.getWindow())) {
    const auto computeTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    auto oldStart = start;
    start = std::chrono::steady_clock::now();
    const auto sleepTime = std::chrono::duration_cast<std::chrono::microseconds>(start - end).count();

    const auto frameTime = std::chrono::duration_cast<std::chrono::microseconds>(start - oldStart).count();

    const auto time = computeTime + sleepTime;
    //const auto time = frameTime;

    /*static bool first = false;
    if (first) {
      throwIf(true, "compute time " + std::to_string(computeTime) + " sleep time " + std::to_string(sleepTime) + " frame time " + std::to_string(frameTime));
    }

    if (!first) {
      first = true;
    }*/

    glfwPollEvents();
    // тут нужно проверить инпут
    updateMouseMovement(time, window, camera);
    updateKeyPressing(time, keyMap);

    // простенькая игровая логика будет у меня наверное в шейдерах
    // там будут наверное только повороты предметов (больше мне особ ничего и не требуется)

    /*static glm::vec3 pos = glm::vec3(1.0f, 3.0f, -10.0f);
    static glm::vec3 move = glm::vec3(speed, 0.0f, 0.0f);
    static DIR dirEnum = RIGHT;*/

    //if ((dirEnum == RIGHT || dirEnum == LEFT) && glm::abs(pos.x) > xMax) {
    //  pos.x = pos.x > 0.0f ? glm::min(pos.x, xMax) : glm::max(pos.x, -xMax);
    //  dirEnum = static_cast<DIR>((dirEnum + 1) % DIR_COUNT);
    //}

    //if ((dirEnum == FORWARD || dirEnum == BACKWARD) && glm::abs(pos.z) > zMax) {
    //  pos.z = pos.z > 0.0f ? glm::min(pos.z, zMax) : glm::max(pos.z, -zMax);
    //  dirEnum = static_cast<DIR>((dirEnum + 1) % DIR_COUNT);
    //}

    //switch (dirEnum) {
    //  case RIGHT:
    //  move = glm::vec3(speed, 0.0f, 0.0f);
    //  break;
    //  case LEFT:
    //  move = glm::vec3(-speed, 0.0f, 0.0f);
    //  break;
    //  case FORWARD:
    //  move = glm::vec3(0.0f, 0.0f, speed);
    //  break;
    //  case BACKWARD:
    //  move = glm::vec3(0.0f, 0.0f, -speed);
    //  break;
    //}

    // нужно сделать управление камерой, как?
    // тип изменять положение от нажатий wasd + мышь
    // как это делается в виндовсе? лучше glfw использовать
    camera.calcView();

    //pos += move;
    //const glm::vec3 pos = glm::vec3(1.0f, 0.0f, -3.0f);
    //const glm::vec3 dir = glm::normalize(glm::vec3(1.0f, -1.0f, 1.0f));
    //const glm::vec3 dir = glm::normalize(glm::vec3(0.0f, 0.0f, 1.0f));
    //const glm::vec3 norm = glm::normalize(pos); //glm::radians(45.0f)
    const glm::mat4 persp = glm::perspective(75.0f, float(window.getWidth()) / float(window.getHeight()), 1.0f, 256.0f);

    //throwIf(true, "window width " + std::to_string(window.getWidth()) + " height " + std::to_string(window.getHeight()));

    //glm::mat4 view = glm::lookAt(pos, pos + dir, glm::vec3(0.0f, 1.0f, 0.0f));

    //const glm::vec3 cameraPos = -norm * distance;
    const glm::vec3 cameraPos = glm::vec3(camera.position());
    //const glm::mat4 view = glm::lookAt(cameraPos, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::mat4 view = camera.view();
    const glm::mat4 viewProj = persp * view;
    const glm::mat4 viewProj2 = view * persp;

    // какие то проблемы у меня с перспективой

    // нужно выделить под это отдельную кнопку
    //render.recreatePSO();

    //render.waitForFrame();

    const uint32_t instanceCount = boxBuffer.size() + icosahedronBuffer.size() + coneBuffer.size();

    render.prepareRender(instanceCount, viewProj);
    render.updateSceneData(glm::vec4(cameraPos, 1.0f), viewProj);

    render.nextFrame();
    render.gBufferPart(boxBuffer.size(), icosahedronBuffer.size(), coneBuffer.size());
    render.rayTracingPart();
    render.filterPart();

    end = std::chrono::steady_clock::now();

    render.endFrame();
  }
}

void initScene(GPUBuffer<ComputeData> &boxBuffer, GPUBuffer<ComputeData> &icosahedronBuffer, GPUBuffer<ComputeData> &coneBuffer) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> disPos(-10, 10);
  std::uniform_real_distribution<> disScale(0.5f, 3);

  //glm::vec4 pos = glm::vec4(-2.0f, 0.0f, 0.0f, 1.0f);
  //const glm::vec4 addPos = glm::vec4(0.1f, 0.1f, 0.1f, 0.0f);

  for (uint32_t i = 0; i < boxBuffer.size(); ++i) {
    //pos += addPos;
    const ComputeData data{
      //glm::vec4(disPos(gen), disPos(gen), disPos(gen), 1.0f),
      glm::vec4(0.0f, 2.0f, 0.0f, 1.0f),
      //pos,
      glm::vec4(0.0f, 0.0f, 0.0f, 0.0f),
      //glm::vec4(disScale(gen), disScale(gen), disScale(gen), 0.0f),
      glm::vec4(1.0f, 1.0f, 1.0f, 0.0f),

      glm::mat4(1.0f)
    };

    boxBuffer[i] = data;
  }

  for (uint32_t i = 0; i < icosahedronBuffer.size(); ++i) {
    const ComputeData data{
      //glm::vec4(disPos(gen), disPos(gen), disPos(gen), 1.0f),
      glm::vec4(2.0f, 2.0f, 0.0f, 1.0f),
      glm::vec4(0.0f, 0.0f, 0.0f, 0.0f),
      //glm::vec4(disScale(gen), disScale(gen), disScale(gen), 0.0f),
      glm::vec4(1.0f, 1.0f, 1.0f, 0.0f),

      glm::mat4(1.0f)
    };

    icosahedronBuffer[i] = data;
  }

  for (uint32_t i = 0; i < coneBuffer.size(); ++i) {
    //pos += addPos;
    const ComputeData data{
      //glm::vec4(disPos(gen), disPos(gen), disPos(gen), 1.0f),
      glm::vec4(-2.0f, 2.0f, 0.0f, 1.0f),
      //pos,
      glm::vec4(0.0f, 0.0f, 0.0f, 0.0f),
      //glm::vec4(disScale(gen), disScale(gen), disScale(gen), 0.0f),
      glm::vec4(1.0f, 1.0f, 1.0f, 0.0f),

      glm::mat4(1.0f)
    };

    coneBuffer[i] = data;
  }
}

void setDefaultKeyMap(Camera &camera, std::unordered_map<int, std::function<void(const uint64_t &)>> &keyMap) {
  keyMap[GLFW_KEY_W] = [&] (const uint64_t &time) {
    camera.forward(time);
  };

  keyMap[GLFW_KEY_S] = [&] (const uint64_t &time) {
    camera.backward(time);
  };

  keyMap[GLFW_KEY_A] = [&] (const uint64_t &time) {
    camera.stepleft(time);
  };

  keyMap[GLFW_KEY_D] = [&] (const uint64_t &time) {
    camera.stepright(time);
  };

  keyMap[GLFW_KEY_SPACE] = [&] (const uint64_t &time) {
    camera.upward(time);
  };

  keyMap[GLFW_KEY_LEFT_CONTROL] = [&](const uint64_t &time) {
    camera.downward(time);
  };
}

void updateMouseMovement(const uint64_t &time, const Window &window, Camera &camera) {
  const float mouseSpeed = 0.05f;

  double xpos, ypos;
  glfwGetCursorPos(window.getWindow(), &xpos, &ypos);

  double centerX = double(window.getWidth()) / 2.0, centerY = double(window.getHeight()) / 2.0;
  glfwSetCursorPos(window.getWindow(), centerX, centerY);

  static float horizontalAngle = 0.0f, verticalAngle = 0.0f;

  const float dt = MCS_TO_SEC(time);

  horizontalAngle += mouseSpeed * dt * float(centerX - xpos);
  verticalAngle += mouseSpeed * dt * float(centerY - ypos);

  camera.handleMouseInput(horizontalAngle, verticalAngle);
}

void updateKeyPressing(const uint64_t &time, std::unordered_map<int, std::function<void(const uint64_t &)>> &keyMap) {
  /*for (uint32_t i = 0; i < stackSize; ++i) {
    if (stack[i].state != GLFW_RELEASE) {
      auto itr = keyMap.find(stack[i].key);
      if (itr == keyMap.end()) continue;

      itr->second(time);
    }
  }*/

  //stackSize = 0;

  for (auto &pair : keyMap) {
    if (keysState[pair.first]) {
      pair.second(time);
    }
  }
}