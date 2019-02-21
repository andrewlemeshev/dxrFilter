#include "Window.h"

#define KEY_STACK_SIZE 50

//std::vector<KeyState> stack(KEY_STACK_SIZE);
//uint32_t stackSize = 0;
bool keysState[GLFW_KEY_LAST + 1];

void throwIfFailed(const HRESULT &hr, const std::string &string) {
  if (FAILED(hr)) throw std::runtime_error(string);
}

void throwIf(const bool &var, const std::string &string) {
  if (var) throw std::runtime_error(string);
}

void error_callback(int error, const char* description) {
  throwIf(true, std::string(description));
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
  /*if (stackSize >= KEY_STACK_SIZE) return;

  stack[stackSize] = {key, action};
  ++stackSize;*/

  keysState[key] = action != GLFW_RELEASE;

  //if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) glfwSetWindowShouldClose(window, GLFW_TRUE); 
}

void mouse_callback(GLFWwindow* window, int button, int action, int mods) {
  keysState[button] = action != GLFW_RELEASE;
}

Window::Window() {
  throwIf(!glfwInit(), "Could not init glfw");

  glfwSetErrorCallback(error_callback);
}

Window::~Window() {
  glfwDestroyWindow(window);

  glfwTerminate();
}

void Window::init(HINSTANCE hInstance, const int showWnd, const uint32_t &width,
                  const uint32_t &height, bool fullscreen) {
  this->size.width = width;
  this->size.height = height;
  this->fullscreen = fullscreen;
  this->hInstance = hInstance;

  GLFWmonitor* mon = nullptr;

  if (fullscreen) {
    mon = glfwGetPrimaryMonitor();
    const GLFWvidmode* vidMode = glfwGetVideoMode(mon);
    this->size.width = vidMode->width;
    this->size.height = vidMode->height;

    /*HMONITOR hmon = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
    MONITORINFO mi = {sizeof(mi)};
    GetMonitorInfo(hmon, &mi);

    this->size.width = mi.rcMonitor.right - mi.rcMonitor.left;
    this->size.height = mi.rcMonitor.bottom - mi.rcMonitor.top;*/
  }

  /*WNDCLASSEX wc;

  wc.cbSize = sizeof(WNDCLASSEX);
  wc.style = CS_HREDRAW | CS_VREDRAW;
  wc.lpfnWndProc = WndProc;
  wc.cbClsExtra = NULL;
  wc.cbWndExtra = NULL;
  wc.hInstance = hInstance;
  wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
  wc.hCursor = LoadCursor(NULL, IDC_ARROW);
  wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 2);
  wc.lpszMenuName = NULL;
  wc.lpszClassName = windowName;
  wc.hIconSm = LoadIcon(NULL, IDI_APPLICATION);

  throwIf(!RegisterClassEx(&wc), "Error registering class");

  hwnd = CreateWindowEx(NULL, windowName, windowTitle, WS_OVERLAPPEDWINDOW,
                        CW_USEDEFAULT, CW_USEDEFAULT, this->size.width,
                        this->size.height, NULL, NULL, hInstance, NULL);*/

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  window = glfwCreateWindow(this->size.width, this->size.height, windowTitleC, mon, nullptr);
  throwIf(window == nullptr, "Error creating window");

  hwnd = glfwGetWin32Window(window);
  throwIf(hwnd == nullptr, "Error getting hwnd");

  /*if (fullscreen) {
    SetWindowLong(hwnd, GWL_STYLE, 0);
  }

  ShowWindow(hwnd, showWnd);
  UpdateWindow(hwnd);*/

  WINDOWINFO info;
  GetWindowInfo(hwnd, &info);

  size.x = info.rcWindow.left;
  size.y = info.rcWindow.top;

  glfwSetKeyCallback(window, key_callback);
  glfwSetMouseButtonCallback(window, mouse_callback);
}

void Window::resize(const uint32_t &width, const uint32_t &height) {
  size.width = width;
  size.height = height;

  SetWindowPos(hwnd, nullptr, size.x, size.y, size.width, size.height, 0);
  UpdateWindow(hwnd);
}

void Window::toggleFullscreen() {
  static WINDOWPLACEMENT wpc;
  static LONG HWNDStyle = 0;
  static LONG HWNDStyleEx = 0;

  fullscreen = !fullscreen;

  if (fullscreen) {
    GetWindowPlacement(hwnd, &wpc);
    if (HWNDStyle == 0) HWNDStyle = GetWindowLong(hwnd, GWL_STYLE);
    if (HWNDStyleEx == 0) HWNDStyleEx = GetWindowLong(hwnd, GWL_EXSTYLE);

    LONG NewHWNDStyle = HWNDStyle;
    NewHWNDStyle &= ~WS_BORDER;
    NewHWNDStyle &= ~WS_DLGFRAME;
    NewHWNDStyle &= ~WS_THICKFRAME;

    LONG NewHWNDStyleEx = HWNDStyleEx;
    NewHWNDStyleEx &= ~WS_EX_WINDOWEDGE;

    SetWindowLong(hwnd, GWL_STYLE, NewHWNDStyle | WS_POPUP);
    SetWindowLong(hwnd, GWL_EXSTYLE, NewHWNDStyleEx | WS_EX_TOPMOST);
    ShowWindow(hwnd, SW_SHOWMAXIMIZED);
  } else {
    SetWindowLong(hwnd, GWL_STYLE, HWNDStyle);
    SetWindowLong(hwnd, GWL_EXSTYLE, HWNDStyleEx);
    ShowWindow(hwnd, SW_SHOWNORMAL);
    SetWindowPlacement(hwnd, &wpc);
  }
}

bool Window::isFullscreen() const { return fullscreen; }

uint32_t Window::getWidth() const { return size.width; }
uint32_t Window::getHeight() const { return size.height; }
HWND Window::getHWND() const { return hwnd; }
GLFWwindow* Window::getWindow() const { return window; }

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
  switch (msg) {
    case WM_KEYDOWN:
      if (wParam == VK_ESCAPE) {
        if (MessageBox(0, L"Are you sure you want to exit?", L"Really?",
                       MB_YESNO | MB_ICONQUESTION) == IDYES) {
          DestroyWindow(hwnd);
        }
      }
      return 0;

    case WM_DESTROY:
      PostQuitMessage(0);
      return 0;
  }

  return DefWindowProc(hwnd, msg, wParam, lParam);
}