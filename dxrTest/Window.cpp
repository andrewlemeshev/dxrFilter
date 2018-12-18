#include "Window.h"

void throwIfFailed(const HRESULT &hr, const std::string &string) {
  if (FAILED(hr)) throw std::runtime_error(string);
}

void throwIf(const bool &var, const std::string &string) {
  if (var) throw std::runtime_error(string);
}

Window::Window() {}

Window::~Window() {}

void Window::init(HINSTANCE hInstance, const int showWnd, const uint32_t &width,
                  const uint32_t &height, bool fullscreen) {
  this->size.width = width;
  this->size.height = height;
  this->fullscreen = fullscreen;
  this->hInstance = hInstance;

  if (fullscreen) {
    HMONITOR hmon = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
    MONITORINFO mi = {sizeof(mi)};
    GetMonitorInfo(hmon, &mi);

    this->size.width = mi.rcMonitor.right - mi.rcMonitor.left;
    this->size.height = mi.rcMonitor.bottom - mi.rcMonitor.top;
  }

  WNDCLASSEX wc;

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
                        this->size.height, NULL, NULL, hInstance, NULL);

  throwIf(hwnd == nullptr, "Error creating window");

  if (fullscreen) {
    SetWindowLong(hwnd, GWL_STYLE, 0);
  }

  ShowWindow(hwnd, showWnd);
  UpdateWindow(hwnd);

  WINDOWINFO info;
  GetWindowInfo(hwnd, &info);

  size.x = info.rcWindow.left;
  size.y = info.rcWindow.top;
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