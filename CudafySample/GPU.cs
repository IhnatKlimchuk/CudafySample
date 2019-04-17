using Cudafy;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace CudafySample
{
    public static class GPU
    {
        [Cudafy]
        public static void CalculateNeuralNetwork(GThread thread, float[] a, float[,,] b, float[] c)
        {
            int startIndex = thread.blockIdx.x * Utils.BLOCK_SIZE * Utils.CHUNK_SIZE + thread.threadIdx.x * Utils.CHUNK_SIZE;
            for (int layerIndex = 0; layerIndex < Utils.LAYER_SIZE; layerIndex++)
            {
                for (int i = 0; i < Utils.CHUNK_SIZE; i++)
                {
                    int itemId = startIndex + i;
                    float sum = 0;
                    for (int j = 0; j < Utils.N; j++)
                    {
                        sum += b[layerIndex, itemId, j] * a[j];
                    }
                    c[itemId] = sum;
                }
                thread.SyncThreads();
                for (int i = 0; i < Utils.CHUNK_SIZE; i++)
                {
                    int itemId = startIndex + i;
                    a[itemId] = c[itemId];
                }
                thread.SyncThreads();
            }
        }
    }
}
