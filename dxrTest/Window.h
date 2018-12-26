#ifndef WINDOW_H
#define WINDOW_H

#include "stdafx.h"

#define GLFW_INCLUDE_NONE
#include "GLFW/glfw3.h"
#define GLFW_EXPOSE_NATIVE_WIN32
#include "GLFW/glfw3native.h"

#include <cstdint>
#include <vector>

static const LPCTSTR windowName = L"DX12Sample";
static const LPCTSTR windowTitle = L"DX12Sample";
static const char* windowNameC = "DX12Sample";
static const char* windowTitleC = "DX12Sample";

struct KeyState {
  bool state;
  int mod;
  // int scancode;
};

extern bool keysState[GLFW_KEY_LAST+1];
//extern uint32_t stackSize;

struct Rect2D {
  uint32_t x;
  uint32_t y;
  uint32_t width;
  uint32_t height;
};

// callback function for windows messages
LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

class Window {
 public:
  Window();
  ~Window();

  void init(HINSTANCE hInstance, const int showWnd, const uint32_t &width,
            const uint32_t &height, bool fullscreen);

  void resize(const uint32_t &width, const uint32_t &height);
  void toggleFullscreen();

  bool isFullscreen() const;

  uint32_t getWidth() const;
  uint32_t getHeight() const;
  HWND getHWND() const;
  GLFWwindow* getWindow() const;
 private:
  bool fullscreen;
  Rect2D size;

  HWND hwnd;
  HINSTANCE hInstance;
  GLFWwindow* window = nullptr;
};

#endif  // !WINDOW_H