using System;
using System.Numerics;
using System.Threading.Tasks;

namespace CudafySample
{
    public static class CPU
    {
        public static void CalculateNeuralNetwork(float[] input, float[][][] NN, float[] output)
        {
            for (int layerIndex = 0; layerIndex < Utils.LAYER_SIZE; layerIndex++)
            {
                for (int i = 0; i < Utils.N; i++)
                {
                    float sum = 0;
                    for (int j = 0; j < Utils.N; j++)
                    {
                        sum += NN[layerIndex][i][j] * input[j];
                    }
                    input[i] = sum;
                }
            }
            for (int i = 0; i < Utils.N; i++)
            {
                output[i] = input[i];
            }
        }

        public static void CalculateNeuralNetwork_Parallel(float[] input, float[][][] NN, float[] output)
        {
            for (int layerIndex = 0; layerIndex < Utils.LAYER_SIZE; layerIndex++)
            {
                Parallel.For(0, Utils.N, i => 
                {
                    float sum = 0;
                    for (int j = 0; j < Utils.N; j++)
                    {
                        sum += NN[layerIndex][i][j] * input[j];
                    }
                    output[i] = sum;
                });
                for (int i = 0; i < Utils.N; i++)
                {
                    input[i] = output[i];
                }
            }
            for (int i = 0; i < Utils.N; i++)
            {
                output[i] = input[i];
            }
        }

        public static void CalculateNeuralNetwork_Vector(float[] input, float[][][] NN, float[] output)
        {
            int vecSize = Vector<float>.Count;
            for (int layerIndex = 0; layerIndex < Utils.LAYER_SIZE; layerIndex++)
            {
                Array.Clear(output, 0, Utils.N);
                for (int i = 0; i < Utils.N; i += vecSize)
                {
                    Vector<float> input_vector = new Vector<float>(input, i);
                    for (int j = 0; j < Utils.N; j++)
                    {
                        Vector<float> partial_vector = new Vector<float>(NN[layerIndex][j], i);
                        var result = input_vector * partial_vector;
                        //for (int k = 0; k < vecSize; k++)
                        //{
                        //    sum += result[k];
                        //}
                        output[j] = result[0] + result[1] + result[2] + result[3];
                    }
                }
                Array.Copy(output, input, Utils.N);
            }
        }

        public static void CalculateNeuralNetwork_Parallel_Vector(float[] input, float[][][] NN, float[] output)
        {
            int vecSize = Vector<float>.Count;
            for (int layerIndex = 0; layerIndex < Utils.LAYER_SIZE; layerIndex++)
            {
                Array.Clear(output, 0, Utils.N);
                Parallel.For(0, Utils.N / vecSize, i =>
                {
                    int startIndex = i * vecSize;
                    Vector<float> input_vector = new Vector<float>(input, i * vecSize);
                    for (int j = 0; j < Utils.N; j++)
                    {
                        Vector<float> partial_vector = new Vector<float>(NN[layerIndex][j], startIndex);
                        var result = input_vector * partial_vector;
                        //for (int k = 0; k < vecSize; k++)
                        //{
                        //    output[j] += result[k];
                        //}
                        output[j] = result[0] + result[1] + result[2] + result[3];
                    }
                });
                Array.Copy(output, input, Utils.N);
            }
        }
    }
}