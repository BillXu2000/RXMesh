#pragma once
#include <helper_cuda.h>
#include <helper_math.h>

struct float33 {
    float3 x, y, z;

    float3& __host__ __device__ operator () (int i) {
        return ((float3*)this)[i];
    }

    float& __host__ __device__ operator () (int i, int j) {
        return ((float*)this)[i * 3 + j];
    }

    float33 __host__ __device__ operator + (float33 k) {
        return float33{x + k.x, y + k.y, z + k.z};
    }

    float33 __host__ __device__ operator - (float33 k) {
        return float33{x - k.x, y - k.y, z - k.z};
    }

    float33 __host__ __device__ operator * (float33 k) {
        return float33{x * k.x, y * k.y, z * k.z};
    }

    float33 __host__ __device__ operator * (float k) {
        return float33{x * k, y * k, z * k};
    }
};

float33 __host__ __device__ transpose(float33 m) {
    return float33{m(0, 0), m(1, 0), m(2, 0), m(0, 1), m(1, 1), m(2, 1), m(0, 2), m(1, 2), m(2, 2)};
}

float __host__ __device__ determinant(float33 m) {
    /*float ans = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) if (i != j) {
            int k = 3 - i - j;
            ans += m(0, i) * m(1, j) * m(2, k) * ((i > j) ^ (i > k) ^ (j > k) ? -1 : 1);
        }
    }*/
    typedef float T;
    T *x = (float*)&m;
    T cofactor11=x[4]*x[8]-x[7]*x[5],cofactor12=x[7]*x[2]-x[1]*x[8],cofactor13=x[1]*x[5]-x[4]*x[2];
    T ans=x[0]*cofactor11+x[3]*cofactor12+x[6]*cofactor13;
    return ans;
}

float33 __host__ __device__ inverse(float33 m) {
    /*float33 ans = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (int i = 0; i < 3; i++) {
        auto swap = [&](float3 &i, float3 &j) {
            float3 k = i;
            i = j;
            j = k;
        };
        for (int j = i + 1; j < 3; j++) if (abs(m(j, i)) > abs(m(i, i))) {
            swap(m(i), m(j));
            swap(ans(i), ans(j));
        }
        ans(i) /= m(i, i);
        m(i) /= m(i, i);
        for (int j = i + 1; j < 3; j++) {
            ans(j) -= ans(i) * m(j, i);
            m(j) -= m(i) * m(j, i);
        }
    }
    for (int i = 2; i >= 0; i--) {
        for (int j = 0; j < i; j++) {
            ans(j) -= ans(i) * m(j, i);
            m(j) -= m(i) * m(j, i);
        }
    }
    return ans;*/
    typedef float T;
    float33 ans;
    T *x = (float*)&m, *inv = (float*)&ans;
    T cofactor11=x[4]*x[8]-x[7]*x[5],cofactor12=x[7]*x[2]-x[1]*x[8],cofactor13=x[1]*x[5]-x[4]*x[2];
    T determinant=x[0]*cofactor11+x[3]*cofactor12+x[6]*cofactor13;
    T s=1/determinant;
    inv[0]=s*cofactor11; inv[1]=s*cofactor12; inv[2]=s*cofactor13;
    inv[3]=s*x[6]*x[5]-s*x[3]*x[8]; inv[4]=s*x[0]*x[8]-s*x[6]*x[2]; inv[5]=s*x[3]*x[2]-s*x[0]*x[5];
    inv[6]=s*x[3]*x[7]-s*x[6]*x[4]; inv[7]=s*x[6]*x[1]-s*x[0]*x[7]; inv[8]=s*x[0]*x[4]-s*x[3]*x[1];
    return ans;
}

float33 __host__ __device__ mul(float33 _a, float33 _b) {
    /*float33 ans;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            ans(i, j) = 0;
            for (int k = 0; k < 3; k++) {
                ans(i, j) += a(i, k) * b(k, j);
            }
        }
    }*/
    float33 ans;
    float *a = reinterpret_cast<float*>(&_b), *b = reinterpret_cast<float*>(&_a), *c = reinterpret_cast<float*>(&ans);
    c[0]=a[0]*b[0]+a[3]*b[1]+a[6]*b[2]; 
    c[1]=a[1]*b[0]+a[4]*b[1]+a[7]*b[2];
    c[2]=a[2]*b[0]+a[5]*b[1]+a[8]*b[2];
    c[3]=a[0]*b[3]+a[3]*b[4]+a[6]*b[5];
    c[4]=a[1]*b[3]+a[4]*b[4]+a[7]*b[5];
    c[5]=a[2]*b[3]+a[5]*b[4]+a[8]*b[5];
    c[6]=a[0]*b[6]+a[3]*b[7]+a[6]*b[8];
    c[7]=a[1]*b[6]+a[4]*b[7]+a[7]*b[8];
    c[8]=a[2]*b[6]+a[5]*b[7]+a[8]*b[8];
    return ans;
}

float3 __host__ __device__ mul(float33 _a, float3 _b) {
    /*float33 ans;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            ans(i, j) = 0;
            for (int k = 0; k < 3; k++) {
                ans(i, j) += a(i, k) * b(k, j);
            }
        }
    }*/
    float3 ans;
    float *a = reinterpret_cast<float*>(&_a), *b = reinterpret_cast<float*>(&_b), *c = reinterpret_cast<float*>(&ans);
    c[0]=a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; 
    c[1]=a[3]*b[0]+a[4]*b[1]+a[5]*b[2];
    c[2]=a[6]*b[0]+a[7]*b[1]+a[8]*b[2];
    return ans;
}