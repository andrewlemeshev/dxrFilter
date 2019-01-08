#ifndef BUFFER_H
#define BUFFER_H

#include <cstdint>
#include <cstddef>

#include <dxgi1_4.h>
#include <d3d12.h>
#include <D3Dcompiler.h>
//#include <DirectXMath.h>
#include "d3dx12.h"

// нужно ли мне от него наследовать?  
template <typename T>
class GPUBuffer {
public:
  const float resizePolicy = 1.5f;
  const size_t defaultSize = 20;

  GPUBuffer() {}
  ~GPUBuffer() {
    destroy();
  }

  void construct(ID3D12Device* device) {
    if (isValid()) destroy();

    this->device = device;
    m_capacity = defaultSize;

    create(m_capacity);
  }

  void construct(ID3D12Device* device, const size_t &size) {
    if (isValid()) destroy();

    this->device = device;
    m_capacity = size;
    m_size = size;

    create(m_capacity);

    for (size_t i = 0; i < m_size; ++i) {
      ptr[i] = T();
    }
  }

  void destroy() {
    if (!isValid()) return;

    buffer->Unmap(0, nullptr);
    buffer->Release();

    ptr = nullptr;
    m_size = 0;
    m_capacity = 0;

    buffer = nullptr;
    device = nullptr;
  }

  bool isValid() const {return device != nullptr; }
  size_t size() const { return m_size; }
  size_t capasity() const { return m_capacity; }
  size_t buffer_size() const { return m_capacity * sizeof(T); }

  T* data() { return ptr; }
  const T* data() const { return ptr; }

  T & at(const size_t &index) { return ptr[index]; }
  const T & at(const size_t &index) const { return ptr[index]; }

  T & operator[] (const size_t &index) { return ptr[index]; }
  const T & operator[] (const size_t &index) const { return ptr[index]; }

  void push_back(const T &value) {
    if (m_size == m_capacity) recreate(m_capacity * resizePolicy);

    ptr[m_size] = value;
    ++m_size;
  }

  void resize(const size_t &size) {
    if (m_size == size) return;

    if (size == 0) {
      clear();
      return;
    }

    if (size > m_size) {
      if (size > m_capacity) reserve(size);

      for (size_t i = m_size; i < size; ++i) {
        ptr[i] = T();
      }

      m_size = size;
      return;
    }

    for (size_t i = size; i < m_size; ++i) {
      ptr[i].~T();
    }

    m_size = size;
  }

  void reserve(const size_t &capasity) {
    if (m_capacity >= capasity) return;

    recreate(capasity);
  }

  void clear() {
    for (size_t i = 0; i < m_size; ++i) {
      ptr[i].~T();
    }

    m_size = 0;
  }
private:
  T* ptr = nullptr;
  size_t m_size = 0;
  size_t m_capacity = 0;

  // тут явно потребуется что то еще
  // тут потребуется буфер вьев
  ID3D12Resource* buffer = nullptr;
  ID3D12Device* device = nullptr;

  void recreate(const size_t &newCapacity) {
    T* oldPtr = ptr;
    const size_t oldCapacity = m_capacity;
    ID3D12Resource* oldBuffer = buffer;
    
    create(newCapacity);

    memcpy(ptr, oldPtr, oldCapacity * sizeof(T));

    oldBuffer->Unmap(0, nullptr);
    oldBuffer->Release();

    // это все?
  }

  void create(const size_t &newCapacity) {
    const size_t bufferSize = newCapacity * sizeof(T);

    HRESULT hr;
    hr = device->CreateCommittedResource(
      &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD), // upload heap
      D3D12_HEAP_FLAG_NONE, // no flags
      &CD3DX12_RESOURCE_DESC::Buffer(bufferSize), // resource description for a buffer
      D3D12_RESOURCE_STATE_GENERIC_READ, // GPU will read from this buffer and copy its contents to the default heap
      nullptr,
      IID_PPV_ARGS(&buffer));

    if (FAILED(hr)) {
      MessageBox(0, L"Failed to create GPUBuffer", L"Error", MB_OK);
      return;
    }

    void* data;
    buffer->Map(0, nullptr, &data);
    ptr = (T*)data;

    m_capacity = newCapacity;
  }
};

#endif // !BUFFER_H
