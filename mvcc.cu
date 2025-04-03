#include <iostream>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <climits>

struct MVCCVersion {
    int value;
    int begin_ts;
    int end_ts;
    bool is_committed;
};

class MVCCStore {
private:
    std::unordered_map<int, std::vector<MVCCVersion>> versions;
    std::mutex mutex;
    int current_timestamp;

public:
    MVCCStore() : current_timestamp(0) {}
    void write(int object_id, int value) {
        std::lock_guard<std::mutex> lock(mutex);
        int ts = ++current_timestamp;
        if (!versions[object_id].empty()) {
            versions[object_id].back().end_ts = ts;
            versions[object_id].back().is_committed = true;
        }
        MVCCVersion new_version;
        new_version.value = value;
        new_version.begin_ts = ts;
        new_version.end_ts = INT_MAX;
        new_version.is_committed = false;
        versions[object_id].push_back(new_version);
    }

    int read(int object_id) {
        std::lock_guard<std::mutex> lock(mutex);
        if (versions.find(object_id) == versions.end() || versions[object_id].empty()) { return -1; }
        for (auto it = versions[object_id].rbegin(); it != versions[object_id].rend(); ++it) { if (it->is_committed) { return it->value; } }
        return -1;
    }

    int read_at_ts(int object_id, int ts) {
        std::lock_guard<std::mutex> lock(mutex);
        if (versions.find(object_id) == versions.end()) {
            return -1;
        }
        for (const auto& version : versions[object_id]) { if (version.begin_ts <= ts && ts < version.end_ts && version.is_committed) { return version.value; }}
        return -1;
    }
    void commit() {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto& [object_id, object_versions] : versions) { if (!object_versions.empty() && !object_versions.back().is_committed) { object_versions.back().is_committed = true; } }
    }
    void rollback() {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto& [object_id, object_versions] : versions) { if (!object_versions.empty() && !object_versions.back().is_committed) { object_versions.pop_back(); } }
    }
};

int main() {
    // Testing single writes
    MVCCStore store;
    store.write(1, 100);
    store.commit();
    store.write(1, 200);
    store.commit();
    store.write(1, 300);
    std::cout << "Looking at object 1" << std::endl;
    std::cout << "Latest value (committed): " << store.read(1) << std::endl;
    std::cout << "Value at timestamp 1: " << store.read_at_ts(1, 1) << std::endl;
    std::cout << "Value at timestamp 2: " << store.read_at_ts(1, 2) << std::endl;
    std::cout << "Value at timestamp 3: " << store.read_at_ts(1, 3) << std::endl;
    std::cout << "Value at timestamp 4: " << store.read_at_ts(1, 4) << std::endl;
    return 0;
} 