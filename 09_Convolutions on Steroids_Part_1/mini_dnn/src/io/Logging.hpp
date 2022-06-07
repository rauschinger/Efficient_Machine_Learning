#ifndef MINI_DNN_IO_LOGGING_HPP
#define MINI_DNN_IO_LOGGING_HPP

#include <cassert>
#include <iostream>

#define MINI_DNN_LOG_INFO std::cout << ""

namespace mini_dnn {
  namespace io {
    class NullStream {
      public:
        NullStream() {}

        template<typename T>
        NullStream& operator<<( T const& ) {
          return *this;
        }
    };
  }
}

#define MINI_DNN_CHECK(str)           assert( str          ); mini_dnn::io::NullStream() << ""
#define MINI_DNN_CHECK_EQ(str1, str2) assert( str1 == str2 ); mini_dnn::io::NullStream() << ""
#define MINI_DNN_CHECK_NE(str1, str2) assert( str1 != str2 ); mini_dnn::io::NullStream() << ""
#define MINI_DNN_CHECK_LT(str1, str2) assert( str1 <  str2 ); mini_dnn::io::NullStream() << ""
#define MINI_DNN_CHECK_GT(str1, str2) assert( str1 > str2  ); mini_dnn::io::NullStream() << ""
#define MINI_DNN_CHECK_LE(str1, str2) assert( str1 <= str2 ); mini_dnn::io::NullStream() << ""
#define MINI_DNN_CHECK_GE(str1, str2) assert( str1 >= str2 ); mini_dnn::io::NullStream() << ""

#endif