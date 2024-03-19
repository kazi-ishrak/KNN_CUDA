#include<stdio.h>
#include<iostream>
#include <cuda_runtime.h>
#include<fstream>
#include<string>
#include<math.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

using namespace std;


__global__ void cal_distance(int i, int n, int* red, int* green, int* blue, int* test_red, int* test_green, int* test_blue, int *dis)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n)
    {
        float temp= (red[id] - test_red[i]) * (red[id] - test_red[i]) +
            (green[id] - test_green[i]) * (green[id] - test_green[i]) +
            (blue[id] - test_blue[i]) * (blue[id] - test_blue[i]);

        temp = sqrt(temp)+0.5;
       /* printf("%d ", int(temp));*/
        
        dis[id] = int(temp);
    }
}


void sorting(int n, int k, int* dis, int* res, char* cls) {
    thrust::device_vector<int> distances(dis, dis + n); // Copy distances to device vector
    thrust::device_vector<int> indices(n); // Device vector for indices
    thrust::sequence(thrust::device, indices.begin(), indices.end()); // Initialize indices
    thrust::sort_by_key(distances.begin(), distances.end(), indices.begin()); // Sort by distance

    // Extract the indices of the k nearest neighbors
    for (int x = 0; x < k; x++) {
        res[x] = indices[x];
    }
}

char vote(int k, int res[], char* cls)
{
    int r = 0, g = 0, b = 0, cnt = 1;
    for (int i = 0; i < k; i++)
    {
        
        char ch = cls[res[i]];
        
        if (ch == 'R')
            r++;
        else if (ch == 'G')
            g++;
        else
            b++;



        cnt++;
    }
    if (r == max(r, max(g, b)))
        return 'R';
    else if (g == max(r, max(g, b)))
        return 'G';
    else
        return 'B';

}



int main()
{
    int i, n;
    cout << "Enter the number of reference points:";
    cin >> n;
    cout << endl;

   //host pointers
    int* red, * green, * blue;
    char* cls;
    
   //cuda host mapped memory allocation
   cudaError_t status_red= cudaHostAlloc((void**)&red, n * sizeof(int), cudaHostAllocMapped);
   cudaError_t status_green = cudaHostAlloc((void**)&green, n * sizeof(int), cudaHostAllocMapped);
   cudaError_t status_blue = cudaHostAlloc((void**)&blue, n * sizeof(int), cudaHostAllocMapped);
   cudaError_t status_cls = cudaHostAlloc((void**)&cls, n * sizeof(char), cudaHostAllocMapped);

   //check allocation error
   if (status_red != cudaSuccess)
   {
       cout << "Host allocation failed: " <<cudaGetErrorString(status_red)<< endl;
       return 1;

   }
   if (status_green != cudaSuccess)
   {
       cout << "Host allocation failed: " << cudaGetErrorString(status_green) << endl;
       return 1;
   }
   if (status_blue != cudaSuccess)
   {
       cout << "Host allocation failed: " << cudaGetErrorString(status_blue) << endl;
       return 1;
   }
   if (status_cls != cudaSuccess)
   {
       cout << "Host allocation failed: " << cudaGetErrorString(status_cls) << endl;
       return 1;
   }
   
   //obtain device pointer
   int* d_red, *d_green, *d_blue;
   char *d_cls;
   cudaHostGetDevicePointer(&d_red, red, 0);
   cudaHostGetDevicePointer(&d_green, green, 0);
   cudaHostGetDevicePointer(&d_blue, blue, 0);
   cudaHostGetDevicePointer(&d_cls, cls, 0);


   ifstream file;
   file.open("final_data.txt");

   if (!file.is_open())
   {
       cout << "Error opening file" << std::endl;
   }
   for (i = 0; i < n * 4; i = i + 4)
   {
       file >> red[i / 4];
       file >> green[i / 4];
       file >> blue[i / 4];
       file >> cls[i / 4];
   }
   file.close();

   int* test_red, * test_green, * test_blue;
   char* test_cls;


   //cuda host mapped memory allocation
   cudaError_t status_test_red = cudaHostAlloc((void**)&test_red, n * sizeof(int), cudaHostAllocMapped);
   cudaError_t status_test_green = cudaHostAlloc((void**)&test_green, n * sizeof(int), cudaHostAllocMapped);
   cudaError_t status_test_blue = cudaHostAlloc((void**)&test_blue, n * sizeof(int), cudaHostAllocMapped);
   cudaError_t status_test_cls = cudaHostAlloc((void**)&test_cls, n * sizeof(char), cudaHostAllocMapped);

   //check allocation error
   if (status_test_red != cudaSuccess)
   {
       cout << "Host allocation failed: " << cudaGetErrorString(status_test_red) << endl;
       return 1;

   }
   if (status_test_green != cudaSuccess)
   {
       cout << "Host allocation failed: " << cudaGetErrorString(status_test_green) << endl;
       return 1;
   }
   if (status_test_blue != cudaSuccess)
   {
       cout << "Host allocation failed: " << cudaGetErrorString(status_test_blue) << endl;
       return 1;
   }
   if (status_test_cls != cudaSuccess)
   {
       cout << "Host allocation failed: " << cudaGetErrorString(status_test_cls) << endl;
       return 1;
   }


   //obtain device pointer
   int* d_test_red, * d_test_green, * d_test_blue;
   char* d_test_cls;
   cudaHostGetDevicePointer(&d_test_red, test_red, 0);
   cudaHostGetDevicePointer(&d_test_green, test_green, 0);
   cudaHostGetDevicePointer(&d_test_blue, test_blue, 0);
   cudaHostGetDevicePointer(&d_test_cls, test_cls, 0);



   ifstream file2;
   file2.open("test_data.txt");
   int m;
   cout << "Enter the number of test points:";
   cin >> m;
   for (i = 0; i < m * 4; i = i + 4)
   {
       file2 >> test_red[i / 4];
       file2 >> test_green[i / 4];
       file2 >> test_blue[i / 4];
       file2 >> test_cls[i / 4];
   }
   file2.close();
   int k;
   cout << "Choose your value of k:";
   cin >> k;

   for (i = 0; i < m; i++)
   {
       int threadsPerBlock = 256;
       int blocksPerGrid = (n + 255) / 256;

       int *dis;
       cudaError_t status_dis = cudaHostAlloc((void**)&dis, n * sizeof(int), cudaHostAllocMapped);
       if (status_dis != cudaSuccess)
       {
           cout << "Host allocation failed: " << cudaGetErrorString(status_dis) << endl;
           return 1;

       }

       int* d_dis;
       cudaHostGetDevicePointer(&d_dis, dis, 0);
       cal_distance << <blocksPerGrid,threadsPerBlock >> > (i, n, d_red, d_green, d_blue, d_test_red, d_test_green, d_test_blue, d_dis);
       cudaDeviceSynchronize();

       cudaError_t error1 = cudaGetLastError();
       if (error1 != cudaSuccess) {
           printf("CUDA error: %s\n", cudaGetErrorString(error1));
       }





       int* res;
       cudaError_t status_res = cudaHostAlloc((void**)&res, k * sizeof(int), cudaHostAllocMapped);
       if (status_res != cudaSuccess)
       {
           cout << "Host allocation failed: " << cudaGetErrorString(status_res) << endl;
           return 1;

       }
   
       sorting(n, k, dis, res, cls);

       thrust::device_vector<int> distances(dis, dis + n);
       thrust::device_vector<int> indices(n);
       thrust::sequence(thrust::device, indices.begin(), indices.end());
       thrust::sort_by_key(distances.begin(), distances.end(), indices.begin());

       //now vote
       
       cout << vote(k, res, cls) << endl;

       cudaFreeHost(dis);
       cudaFreeHost(res);
   }

   cudaFreeHost(red);
   cudaFreeHost(green);
   cudaFreeHost(blue);
   cudaFreeHost(cls);

   cudaFreeHost(test_red);
   cudaFreeHost(test_green);
   cudaFreeHost(test_blue);
   cudaFreeHost(test_cls);
}