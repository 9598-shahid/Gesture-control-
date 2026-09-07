// gesture_engine.cpp
//
// Performance-critical hot-loop code for the AI Gesture Command Center.
// Exposed to Python via pybind11 as the `gesture_native` module.
//
// Feature vector layout (21 floats), produced by python/feature_extraction.py:
//   [0-4]   finger extended flags (thumb,index,middle,ring,pinky) in {0,1}
//   [5-8]   inter-fingertip distances, normalized by palm size
//            (thumb-index, index-middle, middle-ring, ring-pinky)
//   [9-13]  wrist -> fingertip distances, normalized by palm size (5 values)
//   [14-18] PIP-joint bend angle per finger, normalized to [0,1] (5 values)
//   [19]    pinch distance (thumb tip <-> index tip), normalized
//   [20]    thumb vertical direction: (thumb_tip.y - wrist.y), normalized,
//            negative = thumb points up in image space
//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <deque>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace py = pybind11;

using Vec = std::vector<float>;

static constexpr size_t FEATURE_SIZE = 21;

static float euclidean(const Vec& a, const Vec& b) {
    float sum = 0.0f;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

static float cosine_sim(const Vec& a, const Vec& b) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na <= 1e-9f || nb <= 1e-9f) return 0.0f;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

// ---------------------------------------------------------------------
// RuleClassifier: fixed, deterministic classification for the built-in
// gesture set. Needs no calibration/training data. Fast enough to run
// unconditionally every frame; used as a first-pass check before falling
// back to the (more expensive) KNN search for custom gestures.
// ---------------------------------------------------------------------
class RuleClassifier {
public:
    RuleClassifier(float pinch_threshold = 0.08f,
                    float thumb_dir_threshold = 0.15f)
        : pinch_threshold_(pinch_threshold),
          thumb_dir_threshold_(thumb_dir_threshold) {}

    std::pair<std::string, float> classify(const Vec& f) const {
        if (f.size() < FEATURE_SIZE) {
            throw std::invalid_argument("feature vector must have 21 values");
        }
        float thumb_ext = f[0], index_ext = f[1], middle_ext = f[2],
              ring_ext = f[3], pinky_ext = f[4];
        float pinch_dist = f[19];
        float thumb_dir = f[20];
        float ext_sum = thumb_ext + index_ext + middle_ext + ring_ext + pinky_ext;

        // Pinch takes priority: thumb+index tips close together regardless
        // of other finger states (drag support).
        if (pinch_dist < pinch_threshold_) {
            float conf = 1.0f - (pinch_dist / pinch_threshold_);
            return {"pinch", clamp01(conf)};
        }
        if (ext_sum < 0.5f) {
            return {"fist", 1.0f};
        }
        if (ext_sum > 4.5f) {
            return {"open_palm", 1.0f};
        }
        bool only_thumb = thumb_ext > 0.5f && index_ext < 0.5f &&
                           middle_ext < 0.5f && ring_ext < 0.5f && pinky_ext < 0.5f;
        if (only_thumb) {
            if (thumb_dir < -thumb_dir_threshold_) {
                return {"thumbs_up", clamp01(-thumb_dir / 0.5f)};
            }
            if (thumb_dir > thumb_dir_threshold_) {
                return {"thumbs_down", clamp01(thumb_dir / 0.5f)};
            }
        }
        bool only_index = index_ext > 0.5f && middle_ext < 0.5f &&
                           ring_ext < 0.5f && pinky_ext < 0.5f;
        if (only_index) {
            return {"point", 0.9f};
        }
        bool index_middle = index_ext > 0.5f && middle_ext > 0.5f &&
                             ring_ext < 0.5f && pinky_ext < 0.5f;
        if (index_middle) {
            return {"two_fingers", 0.9f};
        }
        return {"unknown", 0.0f};
    }

private:
    static float clamp01(float v) { return std::max(0.0f, std::min(1.0f, v)); }
    float pinch_threshold_;
    float thumb_dir_threshold_;
};

// ---------------------------------------------------------------------
// KNNClassifier: nearest-neighbor search over user-calibrated samples for
// custom gestures. Distance metric blends normalized Euclidean distance
// with cosine similarity so both magnitude and "shape" of the pose count.
// ---------------------------------------------------------------------
class KNNClassifier {
public:
    explicit KNNClassifier(int k = 5) : k_(k) {}

    void add_sample(const Vec& features, const std::string& label) {
        if (features.size() < FEATURE_SIZE) {
            throw std::invalid_argument("feature vector must have 21 values");
        }
        samples_.push_back(features);
        labels_.push_back(label);
    }

    void clear() {
        samples_.clear();
        labels_.clear();
    }

    size_t sample_count() const { return samples_.size(); }

    std::pair<std::string, float> classify(const Vec& query) const {
        if (samples_.empty()) return {"unknown", 0.0f};

        std::vector<std::pair<float, size_t>> scored; // (distance, idx)
        scored.reserve(samples_.size());
        for (size_t i = 0; i < samples_.size(); ++i) {
            float ed = euclidean(query, samples_[i]);
            float cs = cosine_sim(query, samples_[i]);
            // blended distance: lower is better. (1-cs) in [0,2]
            float dist = ed + (1.0f - cs);
            scored.emplace_back(dist, i);
        }
        int k = std::min<int>(k_, static_cast<int>(scored.size()));
        std::partial_sort(scored.begin(), scored.begin() + k, scored.end());

        // Distance-weighted voting: each of the k neighbors contributes
        // 1/(1+distance) to its label's score, rather than one flat vote.
        // This stops a majority class with merely-nearby samples from
        // outvoting a minority class with an exact/near-exact match.
        std::unordered_map<std::string, float> weight;
        std::unordered_map<std::string, float> best_dist;
        float total_weight = 0.0f;
        for (int i = 0; i < k; ++i) {
            const std::string& lbl = labels_[scored[i].second];
            float d = scored[i].first;
            float w = 1.0f / (1.0f + d);
            weight[lbl] += w;
            total_weight += w;
            auto it = best_dist.find(lbl);
            if (it == best_dist.end() || d < it->second) best_dist[lbl] = d;
        }
        std::string best_label = "unknown";
        float best_weight = -1.0f;
        for (auto& kv : weight) {
            if (kv.second > best_weight) {
                best_weight = kv.second;
                best_label = kv.first;
            }
        }
        float agreement = total_weight > 1e-9f ? best_weight / total_weight : 0.0f;
        float d = best_dist[best_label];
        float closeness = 1.0f / (1.0f + d); // in (0,1]
        float confidence = std::max(0.0f, std::min(1.0f, 0.5f * agreement + 0.5f * closeness));
        return {best_label, confidence};
    }

    // For persistence: python side owns the JSON file, this just exposes
    // raw samples so profile_manager.py can serialize them.
    std::vector<Vec> samples() const { return samples_; }
    std::vector<std::string> labels() const { return labels_; }

private:
    int k_;
    std::vector<Vec> samples_;
    std::vector<std::string> labels_;
};

// ---------------------------------------------------------------------
// TemporalVoter: sliding-window majority vote over recent per-frame labels
// so a single misclassified frame doesn't fire an action, and rapid label
// flapping near a decision boundary gets debounced.
// ---------------------------------------------------------------------
class TemporalVoter {
public:
    TemporalVoter(int window = 6, float min_agreement = 0.6f)
        : window_(window), min_agreement_(min_agreement) {}

    std::string push(const std::string& label) {
        history_.push_back(label);
        while (static_cast<int>(history_.size()) > window_) history_.pop_front();

        std::unordered_map<std::string, int> counts;
        for (auto& l : history_) counts[l]++;
        std::string best;
        int best_count = -1;
        for (auto& kv : counts) {
            if (kv.second > best_count) {
                best_count = kv.second;
                best = kv.first;
            }
        }
        float agreement = static_cast<float>(best_count) / static_cast<float>(history_.size());
        if (agreement >= min_agreement_) {
            return best;
        }
        return "unknown";
    }

    void reset() { history_.clear(); }

private:
    int window_;
    float min_agreement_;
    std::deque<std::string> history_;
};

// ---------------------------------------------------------------------
// OneEuroFilter: low-latency adaptive smoothing for cursor coordinates.
// Classic Casiez et al. algorithm — cheap per-sample, needs to run at full
// camera frame rate with no jitter, hence implemented natively.
// ---------------------------------------------------------------------
class OneEuroFilter1D {
public:
    OneEuroFilter1D(float freq = 30.0f, float min_cutoff = 1.0f,
                     float beta = 0.02f, float d_cutoff = 1.0f)
        : freq_(freq), min_cutoff_(min_cutoff), beta_(beta), d_cutoff_(d_cutoff),
          initialized_(false), x_prev_(0.0f), dx_prev_(0.0f) {}

    float filter(float x, float dt_override = -1.0f) {
        float dt = dt_override > 0.0f ? dt_override : (1.0f / freq_);
        if (!initialized_) {
            x_prev_ = x;
            dx_prev_ = 0.0f;
            initialized_ = true;
            return x;
        }
        float dx = (x - x_prev_) / dt;
        float edx = low_pass(dx, dx_prev_, alpha(dt, d_cutoff_));
        float cutoff = min_cutoff_ + beta_ * std::fabs(edx);
        float ex = low_pass(x, x_prev_, alpha(dt, cutoff));
        x_prev_ = ex;
        dx_prev_ = edx;
        return ex;
    }

    void reset() { initialized_ = false; }

private:
    static float low_pass(float x, float prev, float a) {
        return a * x + (1.0f - a) * prev;
    }
    static float alpha(float dt, float cutoff) {
        float tau = 1.0f / (2.0f * static_cast<float>(M_PI) * cutoff);
        return 1.0f / (1.0f + tau / dt);
    }

    float freq_, min_cutoff_, beta_, d_cutoff_;
    bool initialized_;
    float x_prev_, dx_prev_;
};

class OneEuroFilter2D {
public:
    OneEuroFilter2D(float freq = 30.0f, float min_cutoff = 1.0f, float beta = 0.02f)
        : fx_(freq, min_cutoff, beta), fy_(freq, min_cutoff, beta) {}

    std::pair<float, float> filter(float x, float y) {
        return {fx_.filter(x), fy_.filter(y)};
    }
    void reset() { fx_.reset(); fy_.reset(); }

private:
    OneEuroFilter1D fx_, fy_;
};

PYBIND11_MODULE(gesture_native, m) {
    m.doc() = "Native (C++) hot-loop helpers for the AI Gesture Command Center";
    m.attr("FEATURE_SIZE") = FEATURE_SIZE;

    py::class_<RuleClassifier>(m, "RuleClassifier")
        .def(py::init<float, float>(),
             py::arg("pinch_threshold") = 0.08f,
             py::arg("thumb_dir_threshold") = 0.15f)
        .def("classify", &RuleClassifier::classify, py::arg("features"),
             "Returns (label, confidence) for the built-in gesture set");

    py::class_<KNNClassifier>(m, "KNNClassifier")
        .def(py::init<int>(), py::arg("k") = 5)
        .def("add_sample", &KNNClassifier::add_sample)
        .def("clear", &KNNClassifier::clear)
        .def("sample_count", &KNNClassifier::sample_count)
        .def("classify", &KNNClassifier::classify)
        .def("samples", &KNNClassifier::samples)
        .def("labels", &KNNClassifier::labels);

    py::class_<TemporalVoter>(m, "TemporalVoter")
        .def(py::init<int, float>(), py::arg("window") = 6, py::arg("min_agreement") = 0.6f)
        .def("push", &TemporalVoter::push)
        .def("reset", &TemporalVoter::reset);

    py::class_<OneEuroFilter2D>(m, "OneEuroFilter2D")
        .def(py::init<float, float, float>(),
             py::arg("freq") = 30.0f, py::arg("min_cutoff") = 1.0f, py::arg("beta") = 0.02f)
        .def("filter", &OneEuroFilter2D::filter)
        .def("reset", &OneEuroFilter2D::reset);
}
