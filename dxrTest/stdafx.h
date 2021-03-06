// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#ifndef STDAFX_H
#define STDAFX_H

#include "targetver.h"

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
// Windows Header Files
#include <windows.h>

// C RunTime Header Files
#include <stdlib.h>
#include <malloc.h>
#include <memory.h>
#include <tchar.h>
#include <cassert>
#include <string>

#ifdef _DEBUG 
  #define ASSERT(expr) assert(expr);
#else
  #define ASSERT(expr)
#endif

//void throwIfFailed(const HRESULT &hr, const std::string &string) {
//  if (FAILED(hr)) throw std::runtime_error(string);
//}
//
//void throwIf(const bool &var, const std::string &string) {
//  if (var) throw std::runtime_error(string);
//}

void throwIfFailed(const HRESULT &hr, const std::string &string);
void throwIf(const bool &var, const std::string &string);

#endif