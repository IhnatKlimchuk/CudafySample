using System;

namespace CudafySample
{
    public static class Utils
    {
        public const int BLOCK_SIZE = 12;
        public const int GRID_SIZE = 2;
        public const int CHUNK_SIZE = 10;
        public const int LAYER_SIZE = 1000;

        public const int N = GRID_SIZE * BLOCK_SIZE * CHUNK_SIZE;

        public static float[][][] GenerateRandomMatrix(int layersSize = LAYER_SIZE, int xSize = N, int ySize = N, int randomSeed = 42)
        {
            Random rand = new Random(randomSeed);
            float[][][] result = new float[layersSize][][];
            for (int i = 0; i < layersSize; i++)
            {
                result[i] = new float[xSize][];
                for (int j = 0; j < xSize; j++)
                {
                    result[i][j] = new float[ySize];
                    for (int k = 0; k < ySize; k++)
                    {
                        result[i][j][k] = (float)rand.NextDouble() * (rand.Next(0, 2) == 0 ? -1 : +1);
                    }
                }
            }
            return result;
        }

        public static float[,,] AsSingleDimension(this float[][][] array)
        {
            float[,,] result = new float[array.Length, array[0].Length, array[0][0].Length];
            for (int i = 0; i < array.Length; i++)
            {
                for (int j = 0; j < array[0].Length; j++)
                {
                    for (int k = 0; k < array[0][0].Length; k++)
                    {
                        result[i, j, k] = array[i][j][k];
                    }
                }
            }
            return result;
        }

        public static float[][][] GenerateEMatrix(int layersSize = LAYER_SIZE, int xSize = N, int ySize = N)
        {
            float[][][] result = new float[layersSize][][];
            for (int i = 0; i < layersSize; i++)
            {
                result[i] = new float[xSize][];
                for (int j = 0; j < xSize; j++)
                {
                    result[i][j] = new float[ySize];
                    for (int k = 0; k < ySize; k++)
                    {
                        if (j == k)
                        {
                            result[i][j][k] = 1;
                        }
                        else
                        {
                            result[i][j][k] = 0;
                        }
                    }
                }
            }
            return result;
        }

        public static float[] GenerateRandomVector(int xSize = N, int randomSeed = 42)
        {
            Random rand = new Random(randomSeed);
            float[] result = new float[xSize];
            for (int i = 0; i < xSize; i++)
            {
                result[i] = (float)rand.NextDouble() * (rand.Next(0, 2) == 0 ? -1 : +1);
            }
            return result;
        }

        public static float[] GenerateEVector(int xSize = N)
        {
            float[] result = new float[xSize];
            for (int i = 0; i < xSize; i++)
            {
                result[i] = 1;
            }
            return result;
        }
    }
}