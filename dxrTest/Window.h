#ifndef WINDOW_H
#define WINDOW_H

#include "stdafx.h"

#include <cstdint>

const LPCTSTR windowName = L"DX12Sample";
const LPCTSTR windowTitle = L"DX12Sample";

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

 private:
  bool fullscreen;
  Rect2D size;

  HWND hwnd;
  HINSTANCE hInstance;
};

#endif  // !WINDOW_H