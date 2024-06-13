
#include "popsift.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <pybind11/stl.h>

namespace pps{

PopSiftContext *ctx = nullptr;
std::mutex g_mutex;

PopSiftContext::PopSiftContext() : ps(nullptr){
    popsift::cuda::device_prop_t deviceInfo;
    deviceInfo.set(0, false);
}

PopSiftContext::~PopSiftContext(){
    ps->uninit();
    delete ps;
    ps = nullptr;
}

void PopSiftContext::setup(float peak_threshold, float edge_threshold, bool use_root, float downsampling){
    bool changed = false;
    if (this->peak_threshold != peak_threshold) { this->peak_threshold = peak_threshold; changed = true; }
    if (this->edge_threshold != edge_threshold) { this->edge_threshold = edge_threshold; changed = true; }
    if (this->use_root != use_root) { this->use_root = use_root; changed = true; }
    if (this->downsampling != downsampling) { this->downsampling = downsampling; changed = true; }

    if (changed){
        config.setThreshold(peak_threshold);
        config.setEdgeLimit(edge_threshold);
        config.setNormMode(use_root ? popsift::Config::RootSift : popsift::Config::Classic );
        config.setFilterSorting(popsift::Config::LargestScaleFirst);
        config.setMode(popsift::Config::OpenCV);
        config.setDownsampling(downsampling);
        // config.setOctaves(4);
        // config.setLevels(3);

        if (!ps){
            ps = new PopSift(config,
                        popsift::Config::ProcessingMode::ExtractingMode,
                        PopSift::ByteImages );
        }else{
            ps->configure(config, false);
        }
    }
}

PopSift *PopSiftContext::get(){
    return ps;
}

py::object popsift(pyarray_uint8 image,
                 float peak_threshold,
                 float edge_threshold,
                 int target_num_features,
                 bool use_root,
                 float downsampling) {
    py::gil_scoped_release release;

    if (!image.size()) return py::none();

    if (!ctx) ctx = new PopSiftContext();

    int width = image.shape(1);
    int height = image.shape(0);
    int numFeatures = 0;
    
    while(true){
        g_mutex.lock();
        ctx->setup(peak_threshold, edge_threshold, use_root, downsampling);
        std::unique_ptr<SiftJob> job(ctx->get()->enqueue( width, height, image.data() ));
        std::unique_ptr<popsift::Features> result(job->get());
        g_mutex.unlock();

        numFeatures = result->getFeatureCount();

        if (numFeatures >= target_num_features || peak_threshold < 0.0001){
            popsift::Feature* feature_list = result->getFeatures();
            int totalFeatures = 0;
            for (size_t i = 0; i < numFeatures; i++){
                totalFeatures += feature_list[i].num_ori;
            }
            std::vector<std::array<float, 4>> points(totalFeatures);
            std::vector<std::array<float, 128>> desc(totalFeatures);
            size_t index = 0;
            for (size_t i = 0; i < numFeatures; i++){
                popsift::Feature pFeat = feature_list[i];
                for(int oriIdx = 0; oriIdx < pFeat.num_ori; oriIdx++){
                    const popsift::Descriptor* pDesc = pFeat.desc[oriIdx];
                    std::copy(pDesc->features, pDesc->features + 128, std::begin(desc[index])); 
                    points[index] = {
                        std::min<float>(std::round(pFeat.xpos), width - 1),
                        std::min<float>(std::round(pFeat.ypos), height - 1),
                        pFeat.sigma,
                        pFeat.orientation[oriIdx]
                    };
                    index++;
                }
            }
            py::list points_list = py::cast(points);
            py::list desc_list = py::cast(desc);
            return py::make_tuple(points_list, desc_list);
        }else{
           // Lower peak threshold if we don't meet the target
           peak_threshold = (peak_threshold * 2.0) / 3.0;
        }
    }

    // We should never get here
    return py::none();
}

}
