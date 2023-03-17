/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstddef>
#include <utility>
#include <unordered_map>
#include <string>
#include <cassert>

class FileMapping
{
public:

  FileMapping(FileMapping&& other) noexcept
  {
    this->operator=(std::move(other));
  };

  FileMapping& operator=(FileMapping&& other) noexcept
  {
    m_isValid     = other.m_isValid;
    m_fileSize    = other.m_fileSize;
    m_mappingType = other.m_mappingType;
    m_mappingPtr  = other.m_mappingPtr;
    m_mappingSize = other.m_mappingSize;
#ifdef _WIN32
    m_win32.file              = other.m_win32.file;
    m_win32.fileMapping       = other.m_win32.fileMapping;
    other.m_win32.file        = nullptr;
    other.m_win32.fileMapping = nullptr;
#else
    m_unix.file       = other.m_unix.file;
    other.m_unix.file = -1;
#endif
    other.m_isValid    = false;
    other.m_mappingPtr = nullptr;

    return *this;
  }

  FileMapping(const FileMapping&) = delete;
  FileMapping& operator=(const FileMapping& other) = delete;
  FileMapping() {}

  ~FileMapping() { close(); }

  enum MappingType
  {
    MAPPING_READONLY,       // opens existing file for read-only access
    MAPPING_READOVERWRITE,  // creates new file with read/write access, overwriting existing files
  };

  // fileSize only for write access
  bool open(const char* filename, MappingType mappingType, size_t fileSize = 0);
  void close();

  const void* data() const { return m_mappingPtr; }
  void*       data() { return m_mappingPtr; }
  size_t      size() const { return m_mappingSize; }
  bool        valid() const { return m_isValid; }

protected:
  static size_t g_pageSize;

#ifdef _WIN32
  struct
  {
    void* file        = nullptr;
    void* fileMapping = nullptr;
  } m_win32;
#else
  struct
  {
    int file = -1;
  } m_unix;
#endif

  bool        m_isValid  = false;
  size_t      m_fileSize = 0;
  MappingType m_mappingType = MappingType::MAPPING_READONLY;
  void*       m_mappingPtr  = nullptr;
  size_t      m_mappingSize = 0;
};

// convenience types
class FileReadMapping : private FileMapping
{
public:
  bool        open(const char* filename) { return FileMapping::open(filename, MAPPING_READONLY, 0); }
  void        close() { FileMapping::close(); }
  const void* data() const { return m_mappingPtr; }
  size_t      size() const { return m_fileSize; }
  bool        valid() const { return m_isValid; }
};

class FileReadOverWriteMapping : private FileMapping
{
public:
  bool open(const char* filename, size_t fileSize)
  {
    return FileMapping::open(filename, MAPPING_READOVERWRITE, fileSize);
  }
  void   close() { FileMapping::close(); }
  void*  data() { return m_mappingPtr; }
  size_t size() const { return m_fileSize; }
  bool   valid() const { return m_isValid; }
};


struct FileMappingList
{
    struct Entry
    {
        FileReadMapping mapping;
        int64_t              refCount = 1;
    };
    std::unordered_map<std::string, Entry>       m_nameToMapping;
    std::unordered_map<const void*, std::string> m_dataToName;
#ifdef _DEBUG
    int64_t m_openBias = 0;
#endif

    bool open(const char* path, size_t* size, void** data)
    {
#ifdef _DEBUG
        m_openBias++;
#endif

        std::string pathStr(path);

        auto it = m_nameToMapping.find(pathStr);
        if(it != m_nameToMapping.end())
        {
            *data = const_cast<void*>(it->second.mapping.data());
            *size = it->second.mapping.size();
            it->second.refCount++;
            return true;
        }

        Entry entry;
        if(entry.mapping.open(path))
        {
            const void* mappingData = entry.mapping.data();
            *data                   = const_cast<void*>(mappingData);
            *size                   = entry.mapping.size();
            m_dataToName.insert({mappingData, pathStr});
            m_nameToMapping.insert({pathStr, std::move(entry)});
            return true;
        }

        return false;
    }

    void close(void* data)
    {
#ifdef _DEBUG
        m_openBias--;
#endif
        auto itName = m_dataToName.find(data);
        if(itName != m_dataToName.end())
        {
            auto itMapping = m_nameToMapping.find(itName->second);
            if(itMapping != m_nameToMapping.end())
            {
                itMapping->second.refCount--;

                if(!itMapping->second.refCount)
                {
                    m_nameToMapping.erase(itMapping);
                    m_dataToName.erase(itName);
                }
            }
        }
    }

    ~FileMappingList()
    {
#ifdef _DEBUG
        assert(m_openBias == 0 && "open/close bias wrong");
#endif
        assert(m_nameToMapping.empty() && m_dataToName.empty() && "not all opened files were closed");
    }
};

