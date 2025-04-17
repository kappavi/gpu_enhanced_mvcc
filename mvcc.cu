#include <iostream>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <climits>
#include <chrono>
#include <random>
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <string>

#define MAX_OBJECTS 1000
#define MAX_VERSIONS 1000

struct MVCCVersion {
    int value;
    int begin_ts;
    int end_ts;
    bool is_committed;
};

__global__ void parallel_write_kernel(
    int* object_ids,
    int* values,
    int n,
    MVCCVersion* versions,
    int* version_counts,
    int timestamp
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int obj_id = object_ids[idx];
        int val = values[idx];
        int version_idx = atomicAdd(&version_counts[obj_id], 1);
        MVCCVersion new_version;
        new_version.value = val;
        new_version.begin_ts = timestamp;
        new_version.end_ts = INT_MAX;
        new_version.is_committed = false;
        versions[obj_id * MAX_VERSIONS + version_idx] = new_version;
    }
}

void checkCudaError(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
// this is just for checking CUDA errors
#define CHECK_CUDA(call) checkCudaError(call, __FILE__, __LINE__)

class MVCCStore {
private:
    std::unordered_map<int, std::vector<MVCCVersion>> versions;
    std::mutex mutex;
    int current_timestamp;
    MVCCVersion* d_versions;
    int* d_version_counts;
    
public:
    MVCCStore() : current_timestamp(0) {
        cudaMalloc(&d_versions, sizeof(MVCCVersion) * MAX_OBJECTS * MAX_VERSIONS);
        cudaMalloc(&d_version_counts, sizeof(int) * MAX_OBJECTS);
        cudaMemset(d_version_counts, 0, sizeof(int) * MAX_OBJECTS);
    }
    
    ~MVCCStore() {
        cudaFree(d_versions);
        cudaFree(d_version_counts);
    }
    
    void batch_write_gpu(const std::vector<int>& object_ids, const std::vector<int>& values) {
        int n = object_ids.size();
        int* d_object_ids;
        int* d_values;
        
        CHECK_CUDA(cudaMalloc(&d_object_ids, sizeof(int) * n));
        CHECK_CUDA(cudaMalloc(&d_values, sizeof(int) * n));
        CHECK_CUDA(cudaMemcpy(d_object_ids, object_ids.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_values, values.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
        
        int block_size = 256;
        int num_blocks = (n + block_size - 1) / block_size;
        
        parallel_write_kernel<<<num_blocks, block_size>>>(
            d_object_ids, d_values, n, d_versions, d_version_counts, ++current_timestamp
        );
        
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaFree(d_object_ids));
        CHECK_CUDA(cudaFree(d_values));
    }
    
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
        if (versions.find(object_id) == versions.end()) return -1;
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

class Benchmark {
public:
    static void run_cpu_test(MVCCStore& store, int num_operations) {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 999);
        
        for (int i = 0; i < num_operations; i++) {
            store.write(dis(gen), i);
            if (i % 100 == 0) store.commit();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "CPU test with " << num_operations << " operations took " << duration.count() / 1000.0 << "ms" << std::endl;
    }
    
    static void run_gpu_test(MVCCStore& store, int num_operations) {
        std::vector<int> warmup_ids(100);
        std::vector<int> warmup_vals(100);
        for (int i = 0; i < 100; i++) {
            warmup_ids[i] = i;
            warmup_vals[i] = i;
        }
        store.batch_write_gpu(warmup_ids, warmup_vals);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 999);
        
        std::vector<int> object_ids(num_operations);
        std::vector<int> values(num_operations);
        
        for (int i = 0; i < num_operations; i++) {
            object_ids[i] = dis(gen);
            values[i] = i;
        }
        
        CHECK_CUDA(cudaDeviceSynchronize());
        
        auto start = std::chrono::high_resolution_clock::now();
        
        store.batch_write_gpu(object_ids, values);
        store.commit();
        
        CHECK_CUDA(cudaDeviceSynchronize());
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "GPU test with " << num_operations << " operations took " << duration.count() / 1000.0 << "ms" << std::endl;
    }
    
    // this is gonna be for running tests with hardcoded csv data 
    static void run_csv_test(MVCCStore& store, const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }
        
        std::vector<int> object_ids;
        std::vector<int> values;
        
        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string item;
            std::vector<std::string> row;
            
            while (std::getline(ss, item, ',')) {
                row.push_back(item);
            }
            
            if (row.size() >= 2) {
                try {
                    int obj_id = std::stoi(row[0]);
                    int value = std::stoi(row[1]);
                    object_ids.push_back(obj_id);
                    values.push_back(value);
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing line: " << line << std::endl;
                }
            }
        }
        // // ttrying this alternate approach 
        // std::vector<int> object_ids;
        // std::vector<int> values;
        // std::vector<int> timestamps;
        // std::string line;
        // while (std::getline(file, line)) {
        //     std::stringstream ss(line);
        
        
        std::cout << "Loaded " << object_ids.size() << " operations from " << filename << std::endl;
        
        // actually running the cpu test
        auto cpu_start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < object_ids.size(); i++) {
            store.write(object_ids[i], values[i]);
            if (i % 100 == 0) store.commit();
        }
        store.commit();
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
        
        std::cout << "CSV CPU test with " << object_ids.size() << " operations took " 
                  << cpu_duration.count() / 1000.0 << "ms" << std::endl;
        
        // actually running the gpu test
        auto gpu_start = std::chrono::high_resolution_clock::now();
        store.batch_write_gpu(object_ids, values);
        store.commit();
        CHECK_CUDA(cudaDeviceSynchronize());
        auto gpu_end = std::chrono::high_resolution_clock::now();
        auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start);
        
        std::cout << "CSV GPU test with " << object_ids.size() << " operations took " 
                  << gpu_duration.count() / 1000.0 << "ms" << std::endl;
    }
};

int main() {
    MVCCStore store;

    // Basic test
    std::cout << "Running basic test..." << std::endl;
    store.write(1, 100);
    store.commit();
    store.write(1, 200);
    store.commit();
    store.write(1, 300);
    std::cout << "Latest value (committed): " << store.read(1) << std::endl;

    // performance benchmarks
    std::cout << "\nRunning benchmarks..." << std::endl;
    
    std::cout << "\nSmall dataset (1000 operations):" << std::endl;
    Benchmark::run_cpu_test(store, 1000);
    Benchmark::run_gpu_test(store, 1000);
    
    std::cout << "\nMedium dataset (10000 operations):" << std::endl;
    Benchmark::run_cpu_test(store, 10000);
    Benchmark::run_gpu_test(store, 10000);
    
    std::cout << "\nLarge dataset (100000 operations):" << std::endl;
    Benchmark::run_cpu_test(store, 100000);
    Benchmark::run_gpu_test(store, 100000);
    
    // CSV file tests.
    // probably for small and large GPU will be slower but for very large it'll be faster
    std::cout << "\nRunning CSV file tests..." << std::endl;
    Benchmark::run_csv_test(store, "test_small.csv");
    Benchmark::run_csv_test(store, "test_large.csv");
    std::cout << "\nRunning CSV file test with very large dataset (100000 operations):" << std::endl;
    Benchmark::run_csv_test(store, "test_very_large.csv");
    
    return 0;
} 