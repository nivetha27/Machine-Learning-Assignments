using System;
using System.Text;
using System.IO;
using System.Collections.Generic;

namespace Assignment3
{
  class Program
  {
    public static string filePath = @"C:\Users\nsathya\Desktop\CSE546P-ML\Machine-Learning-Assignments\Assignment 3\MNIST_PCA\";
    public static string trainingLabelFile = filePath + "train-labels.idx1-ubyte";
    public static string trainingImagesFile = filePath + "train-images-pca.idx2-double";
    public static string testLabelFile = filePath + "t10k-labels.idx1-ubyte";
    public static string testImagesFile = filePath + "t10k-images-pca.idx2-double";
    public static string summaryFile = filePath + "summary_" + DateTime.Now.ToString("dd_MM_HH_mm") + ".csv";
    public static string predictedOutputFile = filePath + "predictedOutput_" + DateTime.Now.ToString("dd_MM_HH_mm") + ".csv";

    public static decimal[][] trainImages;
    public static decimal[][] testImages;
    public static int[] trainLabels;
    public static int[] testLabels;

    static void Main(string[] args)
    {
      try
      {
        Console.WriteLine("\nBegin\n");
        int[] numHiddenNodes = new int[] { 100 }; //{ 10 , 50, 100, 500, 1000 };
        double[] learningRates = new double[] { 0.05 };// { 0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5 };
        double[] momentums = new double[] { 0 };// { 0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5 };
        int numLayers = 3;
        int epochs = 30;

        var data = readImagesAndLabels(trainingImagesFile, trainingLabelFile);
        trainImages = data.Item1;
        trainLabels = data.Item2;
        data = readImagesAndLabels(testImagesFile, testLabelFile);
        testImages = data.Item1;
        testLabels = data.Item2;
        runBackpropogationAlgoUsingBatching(numLayers, epochs, numHiddenNodes, learningRates, momentums, 100);

        // sanityCheckBackPropogationAlgo();
      }
      catch (Exception ex)
      {
        Console.WriteLine(ex.Message);
        Console.ReadLine();
      }
    } // Main method ends

    public static Tuple<decimal[][], int[]> readImagesAndLabels(string imagesFileName, string labelFileName)
    {
      decimal[][] pcaImages = null;
      int[] targetLabels = null;
      FileStream ifsLabels = new FileStream(labelFileName, FileMode.Open);
      FileStream ifsImages = new FileStream(imagesFileName, FileMode.Open);

      BinaryReader brLabels = new BinaryReader(ifsLabels);
      BinaryReader brImages = new BinaryReader(ifsImages);
      while (brImages.BaseStream.Position != brImages.BaseStream.Length)
      {
        int magicImages = Utility.convertBytesToInt(brImages.ReadBytes(sizeof(Int32)));
        int numInputImages = Utility.convertBytesToInt(brImages.ReadBytes(sizeof(Int32)));
        int numInputDimensions = Utility.convertBytesToInt(brImages.ReadBytes(sizeof(Int32)));
        pcaImages = new decimal[numInputImages][];
        for (int i = 0; i < numInputImages; i++)
        {
          pcaImages[i] = new decimal[numInputDimensions];
          for (int j = 0; j < numInputDimensions; j++)
          {
            pcaImages[i][j] = (decimal)Utility.convertBytesToDouble(brImages.ReadBytes(sizeof(Int64)));
          }
        }
      }

      while (brLabels.BaseStream.Position != brLabels.BaseStream.Length)
      {
        int magicLabels = Utility.convertBytesToInt(brLabels.ReadBytes(sizeof(Int32)));
        int numLabels = Utility.convertBytesToInt(brLabels.ReadBytes(sizeof(Int32)));
        targetLabels = new int[numLabels];
        for (int i = 0; i < numLabels; i++)
        {
          byte lbl = brLabels.ReadByte();
          targetLabels[i] = Convert.ToInt32(lbl.ToString());
        }
      }

      ifsImages.Close();
      brImages.Close();
      ifsLabels.Close();
      brLabels.Close();

      return new Tuple<decimal[][], int[]>(pcaImages, targetLabels);
    }
    public static void runBackpropogationAlgoUsingBatching(int numLayers, int epochs, int[] numHiddenNodes, double[] learningRates, double[] momentums, int batchSize = 1) {
      foreach (int hiddenNodes in numHiddenNodes)
      {
        Console.WriteLine("Hidden node count " + hiddenNodes);
        var network = new NeuralNetwork(numLayers, 10, 50, hiddenNodes);
        network.createFeedForwardNetwork();
        foreach (decimal learningRate in learningRates)
        {
          foreach (decimal momentum in momentums)
          {
            for (int epoch = 1; epoch <= epochs; epoch++)
            {
              for (int i = 0; i < trainImages.Length; i++)
              {
                decimal[] targetVector = Utility.convertDigitToVector(trainLabels[i]);
                network.propogateInputForwards(trainImages[i]);
                network.calcErrorForPredictedValForOutputAndHiddenUnits(targetVector);
                network.updateEachNetworkDeltaWeight(learningRate);
                if ((i + 1) % batchSize == 0) {
                  network.updateEachNetworkWeight(batchSize);
                  network.resetEachNetworkDeltaWeight();
                }
                if (i == (trainImages.Length / 2 - 1))
                {
                  Console.WriteLine("At half epoch #" + epoch + " using testing data");
                  predictOutputAndComputeErrors(network.neurons, testImages, testLabels);
                  Console.WriteLine("At half epoch #" + epoch + " using training data");
                  predictOutputAndComputeErrors(network.neurons, trainImages, trainLabels);
                }
              }
              Console.WriteLine("At epoch #" + epoch + " using testing data");
              predictOutputAndComputeErrors(network.neurons, testImages, testLabels);
              Console.WriteLine("At epoch #" + epoch + " using training data");
              predictOutputAndComputeErrors(network.neurons, trainImages, trainLabels);
            }
          }
        }
      }
    }
    public static void predictOutputAndComputeErrors(Neuron[][] network, decimal[][] inputDataSet, int[] targetLabels) {
      var inputNetwork = new NeuralNetwork(network);
      int i = 0;
      int positive = 0;
      decimal mean = 0;
      int predictedDigit;
      foreach (decimal[] data in inputDataSet) {
        inputNetwork.propogateInputForwards(data);
        predictedDigit = inputNetwork.predictedDigit();
        if (predictedDigit == targetLabels[i])
        {
          ++positive;
        }
        mean += inputNetwork.computeSquaredError(Utility.convertDigitToVector(targetLabels[i]));
        ++i;
      }
      mean = mean / (2 * inputDataSet.Length);
      double accuracy = positive * 100.0/ inputDataSet.Length;
      Console.WriteLine("Mean Square Error : " + mean + ", Accuracy : " + accuracy);
    }

    public static void sanityCheckBackPropogationAlgo()
    {
      var network = new NeuralNetwork(3, 2, 2, 2);
      network.createFeedForwardNetwork();
      decimal weight = 0.15M;
      for (int i = 0; i < network.neurons.Length - 1; i++)
      {
        for (int j = 0; j < network.neurons[i].Length; j++)
        {
          network.neurons[i][j].weights[0] = weight;
          for (int k = 1; k < network.neurons[i][j].weights.Length; k++)
          {
            network.neurons[i][j].weights[k] = network.neurons[i][j].weights[k - 1];
            if (j < (network.neurons[i].Length - 1))
            {
              network.neurons[i][j].weights[k] += 0.1M;
            }
          }
          if (j == (network.neurons[i].Length - 2))
          {
            weight += 0.15M;
          }
          else
          {
            weight += 0.05M;
          }
        }
      }
      int batchSize = 1;
      int sampleIndex = 0;
      decimal learningRate = 0.5M;
      network.propogateInputForwards(new decimal[] { 0.05M, 0.1M });
      decimal meanNum = network.computeSquaredError(new decimal[] { 0.01M, 0.99M });
      network.calcErrorForPredictedValForOutputAndHiddenUnits(new decimal[] { 0.01M, 0.99M });
      network.updateEachNetworkDeltaWeight(learningRate);
      if ((sampleIndex + 1) % batchSize == 0)
      {
        network.updateEachNetworkWeight(batchSize);
        network.resetEachNetworkDeltaWeight();
      }
    }
  } // Program class ends
}

