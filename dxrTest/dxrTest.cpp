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

#define MCS_TO_SEC(dt) (float(dt) / 1000000.0f)

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

void imguiInit();
void nextGuiFrame(const Window &window, const uint64_t &time);

struct GuiData {
  ID3D12Resource* performanceBuffer;
  size_t timeStampCount;
  uint64_t frequency;
  FilterConstantBuffer* filterBufferData;
};
void defaultGui(const GuiData &data, DX12Render &render);

//int APIENTRY wWinMain(HINSTANCE hInstance, HINSTANCE, LPWSTR, int nCmdShow) {
int __stdcall wWinMain(HINSTANCE hInstance, HINSTANCE, LPWSTR, int nCmdShow) {
  Window window;
  window.init(hInstance, nCmdShow, width, height, fullScreen);

  DX12Render render;
  render.init(window.getHWND(), window.getWidth(), window.getHeight(), window.isFullscreen());
  //render.recreatePSO();

  ImGui::CreateContext();
  imguiInit();

  GPUBuffer<ComputeData> boxBuffer;
  boxBuffer.construct(render.getDevice(), boxCount);
  GPUBuffer<ComputeData> icosahedronBuffer;
  icosahedronBuffer.construct(render.getDevice(), icosahedronCount);
  GPUBuffer<ComputeData> coneBuffer;
  coneBuffer.construct(render.getDevice(), coneCount);

  initScene(boxBuffer, icosahedronBuffer, coneBuffer);

  const uint32_t instanceCount = boxBuffer.size() + icosahedronBuffer.size() + coneBuffer.size();
  render.prepareRender(instanceCount, glm::mat4(1.0f));
  render.computePartHost(boxBuffer, icosahedronBuffer, coneBuffer);

  render.initRT(window.getWidth(), window.getHeight(), boxBuffer, icosahedronBuffer, coneBuffer);
  render.initFilter(window.getWidth(), window.getHeight());
  render.initImGui();

  mainLoop(window, render, boxBuffer, icosahedronBuffer, coneBuffer);

  return 0;
}

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

  glm::dvec2 cursorPos = glm::dvec2(double(window.getWidth() / 2.0f), double(window.getHeight() / 2.0f));

  bool focusOnInterface = false;

  //std::chrono::time_point<std::chrono::steady_clock> start;
  auto start = std::chrono::steady_clock::now();
  auto end = std::chrono::steady_clock::now();

  while (!glfwWindowShouldClose(window.getWindow())) {
    const auto computeTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    render.waitForRenderContext();

    // glfwGetKey(window.getWindow(), GLFW_KEY_R) == GLFW_PRESS
    if (keysState[GLFW_KEY_R]) {
      render.recreatePSO();
    }
    // glfwGetKey(window.getWindow(), GLFW_KEY_LEFT_ALT) == GLFW_PRESS

    static bool lastSate = false;
    if (!lastSate && keysState[GLFW_KEY_LEFT_ALT]) {
      focusOnInterface = !focusOnInterface;

      if (focusOnInterface) {
        glfwSetInputMode(window.getWindow(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);

        glfwSetCursorPos(window.getWindow(), cursorPos.x, cursorPos.y);
      } else {
        glfwSetInputMode(window.getWindow(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        glfwGetCursorPos(window.getWindow(), &cursorPos.x, &cursorPos.y);
        glfwSetCursorPos(window.getWindow(), double(window.getWidth() / 2.0f), double(window.getHeight() / 2.0f));
      }
    }
    lastSate = keysState[GLFW_KEY_LEFT_ALT];

    auto oldStart = start;
    start = std::chrono::steady_clock::now();
    const auto sleepTime = std::chrono::duration_cast<std::chrono::microseconds>(start - end).count();

    const auto frameTime = std::chrono::duration_cast<std::chrono::microseconds>(start - oldStart).count();

    const auto time = computeTime + sleepTime;

    glfwPollEvents();

    nextGuiFrame(window, time);

    // тут нужно проверить инпут
    if (!focusOnInterface) updateMouseMovement(time, window, camera);
    updateKeyPressing(time, keyMap);
    camera.calcView();

    const glm::mat4 persp = glm::perspective(75.0f, float(window.getWidth()) / float(window.getHeight()), 1.0f, 256.0f);

    const glm::vec3 cameraPos = glm::vec3(camera.position());
    const glm::mat4 view = camera.view();
    const glm::mat4 viewProj = persp * view;
    const glm::mat4 viewProj2 = view * persp;

    const GuiData data{
      render.getTimeStampResource(),
      render.getTimeStampCount(),
      render.getTimeStampFrequency(),
      render.constantBufferData()
    };
    defaultGui(data, render);

    const uint32_t instanceCount = boxBuffer.size() + icosahedronBuffer.size() + coneBuffer.size();

    render.prepareRender(instanceCount, viewProj);
    render.updateSceneData(glm::vec4(cameraPos, 1.0f), viewProj);

    int x, y, width, height;
    glfwGetWindowPos(window.getWindow(), &x, &y);
    glfwGetWindowSize(window.getWindow(), &width, &height);

    ImGui::Render();
    render.renderGui(glm::uvec2(x, y), glm::uvec2(width, height));

    render.nextFrame();
    render.gBufferPart(boxBuffer.size(), icosahedronBuffer.size(), coneBuffer.size());
    render.rayTracingPart();
    render.filterPart();
    render.debugPart();
    render.guiPart();

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

void imguiInit() {
  ImGuiIO& io = ImGui::GetIO();

  io.BackendFlags |= ImGuiBackendFlags_HasMouseCursors;         // We can honor GetMouseCursor() values (optional)
  io.BackendPlatformName = "imgui_impl_win32";
  //io.ImeWindowHandle = hwnd;

  // Keyboard mapping. ImGui will use those indices to peek into the io.KeysDown[] array that we will update during the application lifetime.
  io.KeyMap[ImGuiKey_Tab] = GLFW_KEY_TAB;
  io.KeyMap[ImGuiKey_LeftArrow] = GLFW_KEY_LEFT;
  io.KeyMap[ImGuiKey_RightArrow] = GLFW_KEY_RIGHT;
  io.KeyMap[ImGuiKey_UpArrow] = GLFW_KEY_UP;
  io.KeyMap[ImGuiKey_DownArrow] = GLFW_KEY_DOWN;
  io.KeyMap[ImGuiKey_PageUp] = GLFW_KEY_PAGE_UP;
  io.KeyMap[ImGuiKey_PageDown] = GLFW_KEY_PAGE_DOWN;
  io.KeyMap[ImGuiKey_Home] = GLFW_KEY_HOME;
  io.KeyMap[ImGuiKey_End] = GLFW_KEY_END;
  io.KeyMap[ImGuiKey_Insert] = GLFW_KEY_INSERT;
  io.KeyMap[ImGuiKey_Delete] = GLFW_KEY_DELETE;
  io.KeyMap[ImGuiKey_Backspace] = GLFW_KEY_BACKSPACE;
  io.KeyMap[ImGuiKey_Space] = GLFW_KEY_SPACE;
  io.KeyMap[ImGuiKey_Enter] = GLFW_KEY_ENTER;
  io.KeyMap[ImGuiKey_Escape] = GLFW_KEY_ESCAPE;
  io.KeyMap[ImGuiKey_A] = GLFW_KEY_A;
  io.KeyMap[ImGuiKey_C] = GLFW_KEY_C;
  io.KeyMap[ImGuiKey_V] = GLFW_KEY_V;
  io.KeyMap[ImGuiKey_X] = GLFW_KEY_X;
  io.KeyMap[ImGuiKey_Y] = GLFW_KEY_Y;
  io.KeyMap[ImGuiKey_Z] = GLFW_KEY_Z;
}

void nextGuiFrame(const Window &window, const uint64_t &time) {
  ImGuiIO& io = ImGui::GetIO();
  IM_ASSERT(io.Fonts->IsBuilt() && "Font atlas not built! It is generally built by the renderer back-end. Missing call to renderer _NewFrame() function? e.g. ImGui_ImplOpenGL3_NewFrame().");

  int x, y, width, height;
  glfwGetWindowPos(window.getWindow(), &x, &y);
  glfwGetWindowSize(window.getWindow(), &width, &height);

  // Setup display size (every frame to accommodate for window resizing)
  io.DisplaySize = ImVec2(float(width), float(height));

  // Setup time step
  io.DeltaTime = MCS_TO_SEC(time);

  // Read keyboard modifiers inputs
  io.KeyCtrl = glfwGetKey(window.getWindow(), GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS || 
               glfwGetKey(window.getWindow(), GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS;

  io.KeyShift = glfwGetKey(window.getWindow(), GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS || 
                glfwGetKey(window.getWindow(), GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS;

  io.KeyAlt = glfwGetKey(window.getWindow(), GLFW_KEY_RIGHT_ALT) == GLFW_PRESS || 
              glfwGetKey(window.getWindow(), GLFW_KEY_LEFT_ALT) == GLFW_PRESS;

  io.KeySuper = glfwGetKey(window.getWindow(), GLFW_KEY_RIGHT_SUPER) == GLFW_PRESS || 
                glfwGetKey(window.getWindow(), GLFW_KEY_LEFT_SUPER) == GLFW_PRESS;

  // io.KeysDown[], io.MousePos, io.MouseDown[], io.MouseWheel: filled by the WndProc handler below.

  // Set mouse position
  double xPos, yPos;
  glfwGetCursorPos(window.getWindow(), &xPos, &yPos);
  io.MousePos = ImVec2(xPos, yPos);

  for (uint32_t i = 0; i < 5; ++i) {
    io.MouseDown[i] = keysState[i];
  }

  //// Update OS mouse cursor with the cursor requested by imgui
  //ImGuiMouseCursor mouse_cursor = io.MouseDrawCursor ? ImGuiMouseCursor_None : ImGui::GetMouseCursor();
  //if (g_LastMouseCursor != mouse_cursor) {
  //  g_LastMouseCursor = mouse_cursor;
  //  ImGui_ImplWin32_UpdateMouseCursor();
  //}

  ImGui::NewFrame();
}

const char* names[] = {
  "gBuffer",
  "ray tracing",
  "temporal accumulation",
  "pixel addition",
  "bilateral filter",
  "lightning",
  "tone mapping",
  "resource copy",
  "gui"
};
const size_t namesCount = _countof(names);

void defaultGui(const GuiData &data, DX12Render &render) {
  bool checkBox = true;
  float f = 0.5f;
  float clear_color[3] = {0.0f, 0.0f, 0.0f};
  static uint32_t counter = 0;
  //static bool open = true;

  std::vector<int64_t> stamps(data.timeStampCount);
  int64_t* ptr;
  const D3D12_RANGE range{
    0,
    data.performanceBuffer->GetDesc().Width
  };
  data.performanceBuffer->Map(0, &range, reinterpret_cast<void**>(&ptr));
  memcpy(stamps.data(), ptr, data.timeStampCount*sizeof(int64_t));
  data.performanceBuffer->Unmap(0, nullptr);

  const char* items[] = {"color", "normals", "depths", "shadows", "temporal", "pixelDatas", "bilateral", "lightning"};
  static const char* item_current = items[0];
  static uint32_t currentVariant = 0;

  ImGui::SetNextWindowPos(ImVec2(10, 10));
  ImGui::SetNextWindowSize(ImVec2(390, 270));
  if (ImGui::Begin("Hello, world!", nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::SliderInt("bilateral diameter", reinterpret_cast<int*>(&data.filterBufferData->diameterBilateral), 5, 128);

    ImGui::Separator();

    if (ImGui::BeginCombo("Visualize render target", item_current)) {
      
      for (uint32_t i = 0; i < _countof(items); ++i) {
        static bool isSelected = (item_current == items[i]);
        if (ImGui::Selectable(items[i], isSelected)) {
          item_current = items[i];
        }

        isSelected = (item_current == items[i]);

        if (isSelected) {
          //render.visualize(static_cast<Visualize>(i));
          currentVariant = i;
        }
      }

      ImGui::EndCombo();
    }

    ImGui::Separator();

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

    ImGui::Columns(3);

    for (uint32_t i = 1; i < stamps.size(); ++i) {
      const uint32_t j = i - 1;

      ImGui::Text(("Performance " + std::string(j >= namesCount ? "unnamed part" : names[j]) + " time: ").c_str());
    }

    ImGui::NextColumn();

    for (uint32_t i = 1; i < stamps.size(); ++i) {
      const uint32_t j = i - 1;
      const int64_t time = (std::abs(stamps[i] - stamps[j]) * 1000000) / data.frequency;
      ImGui::Text("%5d", time);
    }

    ImGui::NextColumn();

    for (uint32_t i = 1; i < stamps.size(); ++i) {
      ImGui::Text(" mcs");
    }

    ImGui::SetColumnWidth(0, 290.0f);
    ImGui::SetColumnWidth(1, 50.0f);
    ImGui::SetColumnWidth(2, 40.0f);

    //for (uint32_t i = 1; i < stamps.size(); ++i) {
    //  const uint32_t j = i - 1;

    //  // возьмем время в микросекундах
    //  const int64_t time = (std::abs(stamps[i] - stamps[j]) * 1000000) / data.frequency;
    //  ImGui::PushItemWidth(1000);
    //  ImGui::Text(("Performance " + std::string(j >= namesCount ? "unnamed part" : names[j]) + " time: ").c_str());
    //  //ImGui::PopItemWidth();

    //  ImGui::SameLine();

    //  ImGui::PushItemWidth(300);
    //  ImGui::Text("%d", time);
    //  //ImGui::PopItemWidth();

    //  ImGui::SameLine();

    //  ImGui::PushItemWidth(300);
    //  ImGui::Text(" mcs");
    //  //ImGui::PopItemWidth();
    //}

    //ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
    //ImGui::Checkbox("Some check box", &checkBox);           // Edit bools storing our window open/close state

    //ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
    //ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

    //if (ImGui::Button("Button")) counter++;                 // Buttons return true when clicked (most widgets return true when edited/activated)
  
    //ImGui::SameLine();
    //ImGui::Text("counter = %d", counter);
  }
  ImGui::End();

  render.visualize(static_cast<Visualize>(currentVariant));
}