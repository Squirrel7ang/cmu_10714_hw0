#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

namespace py = pybind11;

template <typename T>
void print_matrix(T* X, int m, int n, std::string name="matrix") {
    std::stringstream ss;

    auto now = std::chrono::system_clock::now();
    auto now_time = std::chrono::system_clock::to_time_t(now);
    ss << std::put_time(std::localtime(&now_time), "%Y-%m-%d %H:%M:%S") << std::endl;
    ss << "name is " << name << std::endl;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            ss << (double) X[i*n+j] << ' ';
        }
        ss << std::endl;
    }
    ss << std::endl;

    std::ofstream file("/home/tang/cs/10714/hw0/src/cpp.log", std::ios::app);  // 追加模式
    if (file) {
        file << ss.str();
        file.close();
    }
}

template <typename T>
void print_info(T msg) {
    std::ofstream file("/home/tang/cs/10714/hw0/src/cpp.log", std::ios::app);  // 追加模式
    if (file) {
        file << msg;
        file.close();
    }
}

template <typename T>
void dot(T* X, T* Y, T* result, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            result[i*p+j] = 0;
            for (int k = 0; k < n; k++) {
                result[i*p+j] += X[i*n+k] * Y[k*p+j];
            }
        }
    }
}

template <typename T>
void transpose(T* X, T* result, int m, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            result[i*m+j] = X[j*n+i];
        }
    }
}

void exp(float* X, float* result, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        result[i] = (float) std::exp((double) X[i]);
    }
}

template <typename T>
void normalize_by_row(T* X, T* result, int m, int n) {
    for (int i = 0; i < m; i++) {
        T sum = 0;
        for (int j = 0; j < n; j++) {
            sum = sum + X[i*n+j];
        }
        for (int j = 0; j < n; j++) {
            result[i*n+j] = X[i*n+j] / sum;
        }
    }
}

template <typename T>
void sub(T* X, T* Y, T* result, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        result[i] = X[i] - Y[i];
    }
}

template <typename S, typename T>
void mul(S factor, T* X, T* result, int m, int n) {
    for (int i = 0; i < m*n; i++) {
        result[i] = factor * X[i];
    }
}

void gen_one_hot(float* result, const unsigned char* position, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        result[i] = 0.0;
    }
    for (int i = 0; i < m; i++) {
        result[i*n + (int)position[i]] = 1.0;
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    float* Z_batch  = new float[batch * k];
    float* Iy_batch = new float[batch * k];
    float* grad     = new float[n     * k];
    float* tmp      = new float[batch * k];
    float* X_batch_T= new float[batch * n];

    for (int i = 0; i < (int)m; i += batch) {
        int batch_size = (i + batch > m) ? (m - i) : batch; 

        const float* X_batch = X + i * n;
        const unsigned char* y_batch = y + i;

        dot<float>((float*)X_batch, theta, Z_batch, batch_size, n, k);
        exp(Z_batch, Z_batch, batch_size, k);
        normalize_by_row<float>(Z_batch, Z_batch, batch_size, k);
        gen_one_hot(Iy_batch, y_batch, batch_size, k);

        // compute grad
        transpose((float*)X_batch, X_batch_T, batch_size, n);
        sub(Z_batch, Iy_batch, tmp, batch_size, k);
        dot(X_batch_T, tmp, grad, n, batch_size, k);
        mul((float)(lr / (float)batch_size), grad, grad, n, k);

        sub(theta, grad, theta, n, k);
    }
    delete[] Z_batch;
    delete[] Iy_batch;
    delete[] grad;
    delete[] tmp;
    delete[] X_batch_T;
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
