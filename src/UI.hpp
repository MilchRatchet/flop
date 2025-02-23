#pragma once

#include "ColorMaps.hpp"
#include "Preview.hpp"
#include <FlopContext.hpp>
#include <GLFW/glfw3.h>
#include <flop/Flop.h>
#include <nfd.h>
#include <string>

class UI
{
public:
    enum class ViewMode
    {
        Source,
        FilteredSource,
        YyCxCz,
        Edge,
        EdgeFiltered,
    };

    void on_scroll(GLFWwindow* window, bool magnify);

    void on_click(GLFWwindow* window, int button, int action);

    void update();

    void render(GLFWwindow* window, VkCommandBuffer cb);

    void mark_viewport_dirty();

    void set_reference(std::string const& reference);

    void set_test(std::string const& test);

    void set_tonemap(Tonemap tonemap);

    void set_exposure(float exposure);

    void set_output(std::string const& output);

    bool analyze(bool issued_from_ui);

private:
    void reset_viewports();

    FlopSummary summary_;
    Preview left_preview_;
    Preview right_preview_;
    Preview error_preview_;
    char const* error_          = nullptr;
    nfdchar_t* reference_path_  = nullptr;
    nfdchar_t* test_path_       = nullptr;
    nfdchar_t* output_path_     = nullptr;
    double last_toggled_        = 0.0;
    float toggle_interval_s_    = 0.75f;
    float error_histogram_[256] = {};
    float error_max_            = 0.f;
    float error_max_occurrence_ = 0.f;
    float error_mean_           = 0.f;
    float exposure_             = 0.f;
    Tonemap tonemap_            = Tonemap::Hable;
    ColorMap color_map_         = ColorMap::Magma;
    bool active_                = false;
    bool viewport_dirty_        = false;
    bool toggled_               = false;
    bool animated_              = false;
    ViewMode view_mode_         = ViewMode::Source;
};
