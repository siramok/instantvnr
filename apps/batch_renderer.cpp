//. ======================================================================== //
//.                                                                          //
//. Copyright 2019-2022 Qi Wu                                                //
//.                                                                          //
//. Licensed under the MIT License                                           //
//.                                                                          //
//. ======================================================================== //
#if defined(_WIN32)
#include <windows.h>
#endif

#include "cmdline.h"
#include <api.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#include <stbi/stb_image.h>
#include <stbi/stb_image_write.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <mutex>
#include <thread>

#include <json/json.hpp>

using json = nlohmann::json;
using namespace vnr::math;

#include <vidi_progress_bar.h>
#include <vidi_highperformance_timer.h>
#include <vidi_logger.h>

using Timer = vidi::details::HighPerformanceTimer;
using Logger = vidi::CsvLogger;

const char *render_modes = " 0 Reference (No Shading)\0"
                           " 1 Reference (Local Illumination)\0"
                           " 2 Reference (Full Shadow)\0"
                           " 3 Reference (Single Shade Heuristic)\0"
                           " 4 Ray Marching (Decoding) \0"
                           " 5 Ray Marching (Sample Streaming) \0"
                           " 6 Ray Marching (In Shader)\0"
                           " 7 Ray Marching + LI (Decoding) \0"
                           " 8 Ray Marching + LI (Sample Streaming) \0"
                           " 9 Ray Marching + LI (In Shader)\0"
                           "10 Ray Marching + SSH (Decoding - Debug) \0"
                           "11 Ray Marching + SSH (Sample Streaming) \0"
                           "12 Ray Marching + SSH (In Shader)\0"
                           "13 Path Tracing (Decoding - Debug) \0"
                           "14 Path Tracing (Sample Streaming) \0"
                           "15 Path Tracing (In Shader)\0";

struct CmdArgs : CmdArgsBase
{
public:
  args::ArgumentParser parser;
  args::HelpFlag help;
  args::Group group_volume;
  args::Group group_camera;
  args::Group required;

  args::ValueFlag<std::string> m_simple_volume;
  args::ValueFlag<std::string> m_neural_volume;
  bool has_simple_volume() { return m_simple_volume; }
  bool has_neural_volume() { return m_neural_volume; }
  std::string volume() { return (m_simple_volume) ? args::get(m_simple_volume) : args::get(m_neural_volume); }

  args::ValueFlag<vec3f, args_impl::Vec3fReader> m_camera_from;
  args::ValueFlag<vec3f, args_impl::Vec3fReader> m_camera_up;
  args::ValueFlag<vec3f, args_impl::Vec3fReader> m_camera_at;
  vec3f camera_from() { return (m_camera_from) ? args::get(m_camera_from) : vec3f(0.f, 0.f, -1000.f); }
  vec3f camera_at() { return (m_camera_at) ? args::get(m_camera_at) : vec3f(0.f, 0.f, 0.f); }
  vec3f camera_up() { return (m_camera_up) ? args::get(m_camera_up) : vec3f(0.f, 1.f, 0.f); }

  args::ValueFlag<std::string> m_tfn;
  std::string tfn() { return args::get(m_tfn); }

  args::ValueFlag<int> m_rendering_mode;
  int rendering_mode() { return (m_rendering_mode) ? args::get(m_rendering_mode) : 0; }

  args::ValueFlag<float> m_sampling_rate;
  float sampling_rate() { return (m_sampling_rate) ? args::get(m_sampling_rate) : 1.f; }

  args::ValueFlag<float> m_density_scale;
  float density_scale() { return (m_density_scale) ? args::get(m_density_scale) : 1.f; }

  args::ValueFlag<int> m_num_frames;
  int num_frames() { return args::get(m_num_frames); }

  args::ValueFlag<std::string> m_expname;
  std::string expname() { return (m_expname) ? args::get(m_expname) : "output"; }

  std::string render_mode_msg()
  {
    std::string msg;
    for (int i = 0; i < Items_Count(render_modes); ++i)
    {
      const char *out_text;
      Items_SingleStringGetter(render_modes, i, &out_text);
      msg += std::string(out_text) + "\n";
    }
    return msg;
  }

public:
  CmdArgs(const char *title, int argc, char **argv)
      : parser(title),
        help(parser, "help", "display the help menu", {'h', "help"}),
        group_volume(parser, "This group is all exclusive:", args::Group::Validators::Xor),
        group_camera(parser, "This group is all or none:", args::Group::Validators::AllOrNone),
        required(parser, "This group is all required:", args::Group::Validators::All),
        m_simple_volume(group_volume, "filename", "the simple volume to render", {"simple-volume"}),
        m_neural_volume(group_volume, "filename", "the neural volume to render", {"neural-volume"}),
        m_camera_from(group_camera, "vec3f", "from where we are looking", {"camera-from"}),
        m_camera_at(group_camera, "vec3f", "which point we are looking at", {"camera-at"}),
        m_camera_up(group_camera, "vec3f", "general up-vector", {"camera-up"}),
        m_tfn(required, "filename", "the transfer function preset", {"tfn"}),
        m_num_frames(required, "int", "number of frames to render", {"num-frames"}),
        m_sampling_rate(parser, "float", "ray marching sampling rate", {"sampling-rate"}),
        m_density_scale(parser, "float", "path tracing density scale", {"density-scale"}),
        m_rendering_mode(parser, "int", render_mode_msg(), {"rendering-mode"}),
        m_expname(parser, "std::string", "experiment name", {"exp"})
  {
    exec(parser, argc, argv);
  }
};

void saveJPG(const std::string &fname, vec2i size, const vec4f *pixels)
{
  std::vector<char> image((uint64_t)size.x * size.y * 4);
  for (uint64_t i = 0; i < (uint64_t)size.x * size.y; ++i)
  {
    const auto in = pixels[i];
    const uint32_t r = (uint32_t)(255.99f * clamp(in.x, 0.f, 1.f));
    const uint32_t g = (uint32_t)(255.99f * clamp(in.y, 0.f, 1.f));
    const uint32_t b = (uint32_t)(255.99f * clamp(in.z, 0.f, 1.f));
    const uint32_t a = (uint32_t)(255.99f * clamp(in.w, 0.f, 1.f));
    image[4 * i + 0] = r;
    image[4 * i + 1] = g;
    image[4 * i + 2] = b;
    image[4 * i + 3] = a;
  }
  stbi_flip_vertically_on_write(1);
  stbi_write_jpg(fname.c_str(), size.x, size.y, 4, image.data(), 100);
}

extern "C" int main(int ac, char **av)
{
  // -------------------------------------------------------
  // Initialize command line arguments.
  // -------------------------------------------------------
  CmdArgs args("Commandline Volume Renderer", ac, av);
  Logger logger;
  logger.initialize({"#", "frame time", "fps"}, args.expname());

  vnrVolume volume;

  if (args.has_simple_volume())
  {
    volume = vnrCreateSimpleVolume(args.volume(), "GPU", false);
  }
  else
  {
    // Use the prefix supplied by --neural-volume to load both files.
    std::string prefix = args.volume(); // e.g., "data/trained/chameleon-20220916154242"
    vnrJson params, network;
    std::string params_file = prefix + "-params.json";
    std::string network_file = prefix + "-network.json";

    std::cout << "Loading parameter file: " << params_file << std::endl;
    try
    {
      vnrLoadJsonBinary(params, params_file);
    }
    catch (const std::exception &e)
    {
      std::cerr << "Error loading parameter file: " << e.what() << std::endl;
      return 1;
    }

    std::cout << "Loading network file: " << network_file << std::endl;
    try
    {
      vnrLoadJsonText(network, network_file);
    }
    catch (const std::exception &e)
    {
      std::cerr << "Error loading network file: " << e.what() << std::endl;
      return 1;
    }

    // Print parsed dims from "macrocell".
    if (params.find("macrocell") != params.end() &&
        params["macrocell"].find("dims") != params["macrocell"].end())
    {
      auto dims = params["macrocell"]["dims"];
      int d0 = dims["x"].get<int>();
      int d1 = dims["y"].get<int>();
      int d2 = dims["z"].get<int>();
      std::cout << "Parsed volume dims: " << d0 << " x " << d1 << " x " << d2 << std::endl;
    }
    else
    {
      std::cerr << "Error: 'macrocell' or 'dims' not found in parameter file!" << std::endl;
      return 1;
    }

    // Ensure the parameter file has a "volume" key with the expected structure.
    if (!params.contains("volume"))
    {
      if (params.contains("macrocell") && params["macrocell"].contains("dims"))
      {
        // Create a new "volume" object that contains only the dims.
        params["volume"] = {{"dims", params["macrocell"]["dims"]}};
        std::cout << "Copied dims to 'volume': " << params["volume"] << std::endl;
      }
      else
      {
        std::cerr << "Error: neither 'volume' nor 'macrocell.dims' found!" << std::endl;
        return 1;
      }
    }

    // Merge network info into the parameters.
    params["network"] = network;

    // Attempt to create the neural volume.
    try
    {
      volume = vnrCreateNeuralVolume(params);
    }
    catch (const std::exception &e)
    {
      std::cerr << "Error creating neural volume: " << e.what() << std::endl;
      if (params.find("volume") != params.end())
        std::cerr << "Volume dims were: " << params["volume"] << std::endl;
      return 1;
    }
  }

  // -------------------------------------------------------
  // Setup camera and transfer function.
  // -------------------------------------------------------
  auto camera = vnrCreateCamera();
  // Set camera using command line values.
  vnrCameraSet(camera, args.camera_from(), args.camera_at(), args.camera_up());
  auto from = vnrCameraGetPosition(camera);
  auto at = vnrCameraGetFocus(camera);
  auto up = vnrCameraGetUpVec(camera);

  auto tfn = vnrCreateTransferFunction(args.tfn());
  vnrTransferFunctionSetValueRange(tfn, range1f(0, 1));

  // -------------------------------------------------------
  // Setup renderer.
  // -------------------------------------------------------
  auto ren = vnrCreateRenderer(volume);
  vnrRendererSetTransferFunction(ren, tfn);
  vnrRendererSetCamera(ren, camera);
  vnrRendererSetFramebufferSize(ren, vec2i(768, 768));
  vnrRendererSetMode(ren, args.rendering_mode());
  vnrRendererSetDenoiser(ren, false);
  vnrRendererSetVolumeDensityScale(ren, args.density_scale());
  vnrRendererSetVolumeSamplingRate(ren, args.sampling_rate());

  // Warm up renders.
  for (int i = 0; i < 5; ++i)
    vnrRender(ren);

  std::vector<double> timings(args.num_frames());
  Timer timer1, timer2;
  timer1.start();
  for (int i = 0; i < args.num_frames(); ++i)
  {
    timer2.reset();
    timer2.start();
    vnrRender(ren);
    timer2.stop();
    timings[i] = timer2.milliseconds();
  }
  timer1.stop();
  const auto totaltime = timer1.milliseconds() / 1000.0;

  for (int i = 0; i < args.num_frames(); ++i)
  {
    logger.log_entry<double>({(double)i, (double)timings[i] / 1000.0, (double)1000.0 / timings[i]});
  }

  const vec4f *pixels = vnrRendererMapFrame(ren);
  saveJPG(args.expname() + "-screenshot.jpg", vec2i(768, 768), pixels);

  std::cout << "Summary: " << args.expname() << std::endl;
  std::cout << "\tvolume: " << args.volume() << std::endl;
  std::cout << "\t   tfn: " << args.tfn() << std::endl;
  std::cout << "\t   fps: " << args.num_frames() / totaltime << std::endl;
  std::cout << "\tdensity scale: " << args.density_scale() << std::endl;
  std::cout << "\tsampling rate: " << args.sampling_rate() << std::endl;
  std::cout << "\tcamera: " << from << std::endl;
  std::cout << "\t        " << at << std::endl;
  std::cout << "\t        " << up << std::endl;

  vnrMemoryQueryPrint("[vnr]");
  return 0;
}
