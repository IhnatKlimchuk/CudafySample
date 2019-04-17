/* 
 * This software is based upon the book CUDA By Example by Sanders and Kandrot
 * and source code provided by NVIDIA Corporation.
 * It is a good idea to read the book while studying the examples!
*/
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using System;
using System.Diagnostics;

namespace CudafySample
{
    public static class Program
    {
        public static void Main()
        {
            Console.WriteLine("Tests for:");
            Console.WriteLine("-CPU");
            Console.WriteLine("-CPU parallel");
            Console.WriteLine("-CPU vector");
            Console.WriteLine("-CPU parallel vector");
            Console.WriteLine("-GPU");

            Console.WriteLine("Let's do it!");
            Console.WriteLine("----------------------------");

            Console.ReadKey();
            Console.WriteLine("Start CPU...");
            var cpu = CallCPU();

            Console.ReadKey();
            Console.WriteLine("Start CPU parallel...");
            var cpu_p = CallCPUParallel();

            Console.ReadKey();
            Console.WriteLine("Start CPU vector...");
            var cpu_v = CallCPUVector();

            Console.ReadKey();
            Console.WriteLine("Start CPU parallel vector...");
            var cpu_pv = CallCPUParallelVector();

            Console.ReadKey();
            Console.WriteLine("Start GPU...");
            var gpu = CallGPU();



            Console.ReadLine();
        }

        public static float[] CallCPU()
        {
            float[] input = Utils.GenerateRandomVector();
            float[][][] NN = Utils.GenerateRandomMatrix();
            float[] output = new float[Utils.N];

            Stopwatch cpuSW = new Stopwatch();
            cpuSW.Start();
            CPU.CalculateNeuralNetwork(input, NN, output);
            cpuSW.Stop();
            Console.WriteLine("CPU: " + cpuSW.ElapsedMilliseconds);
            return output;
        }
        public static float[] CallCPUParallel()
        {
            float[] input = Utils.GenerateRandomVector();
            float[][][] NN = Utils.GenerateRandomMatrix();
            float[] output = new float[Utils.N];

            Stopwatch cpuSW = new Stopwatch();
            cpuSW.Start();
            CPU.CalculateNeuralNetwork_Parallel(input, NN, output);
            cpuSW.Stop();
            Console.WriteLine("CPU_parallel: " + cpuSW.ElapsedMilliseconds);
            return output;
        }
        public static float[] CallCPUVector()
        {
            float[] input = Utils.GenerateRandomVector();
            float[][][] NN = Utils.GenerateRandomMatrix();
            float[] output = new float[Utils.N];

            Stopwatch cpuSW = new Stopwatch();
            cpuSW.Start();
            CPU.CalculateNeuralNetwork_Vector(input, NN, output);
            cpuSW.Stop();
            Console.WriteLine("CPU_vector: " + cpuSW.ElapsedMilliseconds);
            return output;
        }
        public static float[] CallCPUParallelVector()
        {
            float[] input = Utils.GenerateRandomVector();
            float[][][] NN = Utils.GenerateRandomMatrix();
            float[] output = new float[Utils.N];

            Stopwatch cpuSW = new Stopwatch();
            cpuSW.Start();
            CPU.CalculateNeuralNetwork_Parallel_Vector(input, NN, output);
            cpuSW.Stop();
            Console.WriteLine("CPU_parallel_vector: " + cpuSW.ElapsedMilliseconds);
            return output;
        }
        public static float[] CallGPU()
        {
            CudafyModes.Target = eGPUType.OpenCL;
            CudafyModes.DeviceId = 0;
            CudafyTranslator.Language = eLanguage.OpenCL;
            CudafyModule km = CudafyTranslator.Cudafy(ePlatform.Auto, eArchitecture.OpenCL, typeof(GPU));
            GPGPU gpu = CudafyHost.GetDevice(eGPUType.OpenCL, 0);
            gpu.LoadModule(km);
            km.Serialize();

            float[] input = Utils.GenerateRandomVector();
            float[,,] NN = Utils.GenerateRandomMatrix().AsSingleDimension();
            float[] output = new float[Utils.N];

            Stopwatch gpuSW = new Stopwatch();
            gpuSW.Start();
            float[] dev_output = gpu.Allocate<float>(output);
            float[] dev_input = gpu.CopyToDevice(input);
            float[,,] dev_NN = gpu.CopyToDevice(NN);
            gpu.Launch(Utils.GRID_SIZE, Utils.BLOCK_SIZE).CalculateNeuralNetwork(dev_input, dev_NN, dev_output);
            gpu.CopyFromDevice(dev_output, output);
            gpu.FreeAll();
            gpuSW.Stop();
            Console.WriteLine("GPU: " + gpuSW.ElapsedMilliseconds);
            return output;
        }
    }
}