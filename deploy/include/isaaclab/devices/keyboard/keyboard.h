#pragma once

#include <string>
#include <vector>
#include <deque>
#include <termios.h>
#include <unistd.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <iostream>


/**
 * @brief Maintain a keyboard reading thread.
 * And get the latest key value.
 */
class Keyboard
{
public:
  Keyboard()
  {
    if (!isatty(fileno(stdin)) || tcgetattr(fileno(stdin), &_oldSettings) != 0) return;
    _has_terminal = true;
    _newSettings = _oldSettings;
    _newSettings.c_lflag &= (~ICANON & ~ECHO);

    _startKey();

    _thread_running  = true;
    _readThread = std::thread([this] {
      while (_thread_running) {
        if (_running) _read();
        else std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    });
  }

  ~Keyboard()
  {
    _thread_running = false;
    _running = false;
    if (_readThread.joinable()) _readThread.join();
    _pauseKey();
  }

  void update()
  {
    std::lock_guard<std::mutex> lock(_key_mutex);
    if(_key != _last_key)
    {
      on_pressed = _key != "";
      on_released = _key == "";
    }
    else
    {
      on_pressed = false;
      on_released = false;
    }
    
    _last_key = _key;
  }

  /**
   * @brief Get the current key value
   * 
   * @return std::string 
   */
  std::string key() const { std::lock_guard<std::mutex> lock(_key_mutex); return _key; };

  /**
   * @brief Get the String object from keyboard 
   * 
   * @param slogan Used to prompt the user for input
   * @return std::string 
   */
  std::string getString(std::string slogan)
  {
    // Stop reading keyboard value
    _running = false;
    _pauseKey();

    std::string stringtemp;
    std::cout << slogan << std::endl;// prompt
    std::getline(std::cin, stringtemp);

    // Restart reading keyboard value
    _startKey();
    _running = true;

    return stringtemp;
  }

  /**
   * flags; available after update()
   */
  bool on_pressed = false;
  bool on_released = false;

  private:
  std::atomic<bool> _thread_running{false};
  std::atomic<bool> _running{false};
  bool _has_terminal = false;
  mutable std::mutex _key_mutex;
  std::thread _readThread;

  bool read_char(char &c)
  {
    fd_set ready; FD_ZERO(&ready); FD_SET(fileno(stdin), &ready);
    timeval timeout{0, 80000};
    return _thread_running && _running &&
      select(fileno(stdin)+1, &ready, nullptr, nullptr, &timeout) > 0 &&
      read(fileno(stdin), &c, 1) == 1;
  }

  void _read()
  {
    char c; std::string value;
    if (read_char(c)) {
      if (c != '\033') value = c;
      else if (read_char(c) && c == '[' && read_char(c)) {
        switch (c) {
          case 'A': value="up"; break;
          case 'B': value="down"; break;
          case 'C': value="right"; break;
          case 'D': value="left"; break;
        }
      }
    }
    std::lock_guard<std::mutex> lock(_key_mutex);
    _key = value;
  }

  /**
   * @brief Restore keyboard default settings.
   */
  void _pauseKey()
  {
    if (_has_terminal) tcsetattr(fileno(stdin), TCSANOW, &_oldSettings);
    _running = false;
  }

  /**
   * @brief Disable canonical mode and echoing of input characters.
   */
  void _startKey()
  {
    if (_has_terminal) tcsetattr(fileno(stdin), TCSANOW, &_newSettings);
    _running = true;
  }

  std::string _key, _last_key;
  
  termios _oldSettings{}, _newSettings{};
};