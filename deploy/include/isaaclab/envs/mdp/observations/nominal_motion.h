#pragma once
#include <cnpy.h>
#include <cmath>
#include <map>
#include <string>
#include <vector>
#include <stdexcept>

namespace isaaclab {
class NominalMotion {
 public:
  explicit NominalMotion(const std::string& path) {
    auto arrays = cnpy::npz_load(path);
    auto scalar = [&](const std::string& key) {
      const auto& a=arrays.at(key);
      if(a.word_size!=sizeof(float) || a.num_vals!=1 || a.fortran_order) throw std::runtime_error("Invalid nominal scalar");
      return a.data<float>()[0];
    };
    if(scalar("nominal_reference_version")!=1.f || scalar("fps")!=50.f) throw std::runtime_error("Expected v1 50Hz nominal motion");
    for(const auto& item: std::map<std::string,size_t>{{"reference_dof_pos",23},{"reference_xy_vel",2},
        {"reference_yaw_vel",1},{"reference_root_height",1},{"reference_gravity_vector",3},{"reference_foot_contact",2}}) {
      const auto& a=arrays.at(item.first);
      if(a.word_size!=sizeof(float) || a.fortran_order || a.shape.size()!=2 || a.shape[1]!=item.second || a.shape[0]<2)
        throw std::runtime_error("Invalid nominal channel: "+item.first);
      if(frames && frames!=a.shape[0]) throw std::runtime_error("Nominal channel frame mismatch");
      frames=a.shape[0]; widths[item.first]=item.second;
      channels[item.first]=std::vector<float>(a.data<float>(),a.data<float>()+a.num_vals);
      for(float x:channels[item.first]) if(!std::isfinite(x)) throw std::runtime_error("Non-finite nominal motion");
    }
  }
  std::vector<float> at(const std::string& key, size_t frame) const {
    if(frame>=frames) throw std::runtime_error("Nominal frame out of range");
    size_t width=widths.at(key);const auto& data=channels.at(key);
    return {data.begin()+frame*width,data.begin()+(frame+1)*width};
  }
  size_t frames=0;
 private:
  std::map<std::string,size_t> widths;
  std::map<std::string,std::vector<float>> channels;
};
}
