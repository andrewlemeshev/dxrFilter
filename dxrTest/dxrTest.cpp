// dxrTest.cpp : Defines the entry point for the application.
//

#include "stdafx.h"
//#include "dxrTest.h"

#include "Window.h"
#include "Render.h"
#include "Buffer.h"

#include <random>

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

int APIENTRY wWinMain(HINSTANCE hInstance, HINSTANCE, LPWSTR, int nCmdShow){
  // что нам нужно в приложении?
  // несколько буферов хранящие данные объектов
  // собственно сами объекты (бокс, конус, сфера)
  // где мне взять сферу и конус? наверное еще нужен материал (честно говоря даже не понимаю что нужно в этом материале указать)
  // нужна еще отражающая плоскость в качестве пола (ну эт просто)
  // генерировать объекты? было бы неплохо наверное сделать дефолтную сцену какую нибудь
  // как создавать акселлератион стракт? 

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

  while (true) {
    if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
      if (msg.message == WM_QUIT)
        break;

      TranslateMessage(&msg);
      DispatchMessage(&msg);
    } else {
      // простенькая игровая логика будет у меня наверное в шейдерах
      // там будут наверное только повороты предметов (больше мне особ ничего и не требуется)

      const float speed = 0.05f;
      const float xMax = 10.0f;
      const float zMax = 10.0f;
      const float distance = 20.0f;
      static glm::vec3 pos = glm::vec3(1.0f, 3.0f, -10.0f);
      static glm::vec3 move = glm::vec3(speed, 0.0f, 0.0f);
      static DIR dirEnum = RIGHT;

      if ((dirEnum == RIGHT || dirEnum == LEFT) && glm::abs(pos.x) > xMax) {
        pos.x = pos.x > 0.0f ? glm::min(pos.x, xMax) : glm::max(pos.x, -xMax);
        dirEnum = static_cast<DIR>((dirEnum + 1) % DIR_COUNT);
      }

      if ((dirEnum == FORWARD || dirEnum == BACKWARD) && glm::abs(pos.z) > zMax) {
        pos.z = pos.z > 0.0f ? glm::min(pos.z, zMax) : glm::max(pos.z, -zMax);
        dirEnum = static_cast<DIR>((dirEnum + 1) % DIR_COUNT);
      }

      switch (dirEnum) {
        case RIGHT:
          move = glm::vec3( speed, 0.0f, 0.0f);
        break;
        case LEFT:
          move = glm::vec3(-speed, 0.0f, 0.0f);
        break;
        case FORWARD:
          move = glm::vec3( 0.0f,  0.0f, speed);
        break;
        case BACKWARD:
          move = glm::vec3( 0.0f,  0.0f,-speed);
        break;
      }
       
      pos += move;
      //const glm::vec3 pos = glm::vec3(1.0f, 0.0f, -3.0f);
      //const glm::vec3 dir = glm::normalize(glm::vec3(1.0f, -1.0f, 1.0f));
      const glm::vec3 dir = glm::normalize(glm::vec3(0.0f, 0.0f, 1.0f));
      const glm::vec3 norm = glm::normalize(pos); //glm::radians(45.0f)
      const glm::mat4 persp = glm::perspective(75.0f, float(window.getWidth()) / float(window.getHeight()), 1.0f, 256.0f);

      //throwIf(true, "window width " + std::to_string(window.getWidth()) + " height " + std::to_string(window.getHeight()));

      //glm::mat4 view = glm::lookAt(pos, pos + dir, glm::vec3(0.0f, 1.0f, 0.0f));

      const glm::vec3 cameraPos = -norm * distance;
      const glm::mat4 view = glm::lookAt(cameraPos, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
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
      render.endFrame();
    }
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