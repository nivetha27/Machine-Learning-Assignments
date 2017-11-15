using System;
using System.Text;
using System.IO;
using System.Collections.Generic;

namespace Assignment3
{
  class Program
  {
    public static string filePath = @"C:\Users\nsathya\Desktop\CSE546P-ML\Machine-Learning-Assignments\Assignment 3\MNIST_PCA\";
    public static string trainingLabelFile = "train-labels.idx1-ubyte";
    public static string trainingImagesFile = "train-images-pca.idx2-double";
    public static string testLabelFile = "t10k-labels.idx1-ubyte";
    public static string testImagesFile = "t10k-images-pca.idx2-double";
    public static string summaryFile = "summary_" + DateTime.Now.ToString("dd_MM_HH_mm") + ".csv";
    public static string predictedOutputFile = "predictedOutput_" + DateTime.Now.ToString("dd_MM_HH_mm") + ".csv";

    public static decimal[][] trainImages;
    public static decimal[][] testImages;
    public static int[] trainLabels;
    public static int[] testLabels;

    static void Main(string[] args)
    {
      try
      {
        var options = new Options();
        bool isValid = CommandLine.Parser.Default.ParseArgumentsStrict(args, options);
        Console.WriteLine("\nBegin\n");
        int batchSize = 100;
        int[] numHiddenNodes = null;
        decimal[] learningRates = null;
        decimal[] momentums = null;
        int numLayers = 3;
        int epochs = 30;
        if (isValid) {
          if (options.help) {
            Console.WriteLine(CommandLine.Text.HelpText.AutoBuild(options));
            return;
          }

          if(options.numHiddenNodes != null) {
            numHiddenNodes = new int[] { (int)options.numHiddenNodes };
          } else {
            numHiddenNodes = new int[] { 100 , 50, 10, 500, 1000 };
          }

          if(options.learningRate != null) {
            learningRates = new decimal[] { (decimal)options.learningRate };
          } else {
            learningRates = new decimal[] { 0.3M }; // { 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5 };
          }

          if(options.momentum != null) {
            momentums = new decimal[] { (decimal)options.momentum };
          } else {
            momentums =  new decimal[] { 0.3M };// { 0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5 };
          }

          if(options.epochs != null) {
            epochs = (int)options.epochs;
          }

          if(options.batchSize != null) {
            batchSize = (int)options.batchSize;
          }

          if(options.filePath != null) {
            filePath = options.filePath;
          }
        }
        var data = readImagesAndLabels(trainingImagesFile, trainingLabelFile);
        trainImages = data.Item1;
        trainLabels = data.Item2;
        data = readImagesAndLabels(testImagesFile, testLabelFile);
        testImages = data.Item1;
        testLabels = data.Item2;
        runBackpropogationAlgoUsingBatching(numLayers, epochs, numHiddenNodes, learningRates, momentums, batchSize);
        //runBackpropogationAlgoUsingBatching(numLayers, epochs, numHiddenNodes, learningRates, momentums, 10);
        //runBackpropogationAlgoUsingBatching(numLayers, epochs, numHiddenNodes, learningRates, momentums, 100);
        //runBackpropogationAlgoUsingBatching(numLayers, epochs, numHiddenNodes, learningRates, momentums, 1000);
        //runBackpropogationAlgoUsingBatching(numLayers, epochs, numHiddenNodes, learningRates, momentums, 10000);

        //sanityCheckBackPropogationAlgo();
      }
      catch (Exception ex)
      {
        Console.WriteLine(ex.Message);
        Console.ReadLine();
      }
    } // Main method ends

    public static Tuple<decimal[][], int[]> readImagesAndLabels(string imagesFileName, string labelFileName)
    {
      imagesFileName = filePath + imagesFileName;
      labelFileName = filePath + labelFileName;
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
    public static void runBackpropogationAlgoUsingBatching(int numLayers, int epochs, int[] numHiddenNodes, decimal[] learningRates, decimal[] momentums, int batchSize = 1) {
      List<string> output = new List<string>();
      //output.Add("Batch Size, Hidden Nodes Count, Learning Rate, Momentum, Epoch, Test Square Error, Test Accuracy, Train Square Error, Train Accuracy");
      output.Add("Batch Size, Hidden Nodes Count, Learning Rate, Momentum, Epoch, Test Square Error, Test Accuracy");
      foreach (int hiddenNodes in numHiddenNodes)
      {
        foreach (decimal learningRate in learningRates)
        {
          foreach (decimal momentum in momentums)
          {
            Console.WriteLine("Hidden node count : " + hiddenNodes + ", Learning rate : " + learningRate + ", Momentum : " + momentum + ", Batch Size : " + batchSize );
            var network = new NeuralNetwork(numLayers, 10, 50, hiddenNodes);
            network.createFeedForwardNetwork(); 
            int epoch = 1;
            for (; epoch <= epochs; epoch++)
            {
              StringBuilder outputLineHalfEpoch = new StringBuilder();
              StringBuilder outputLineEpoch = new StringBuilder();
              outputLineHalfEpoch.Append(batchSize + ",");
              outputLineEpoch.Append(batchSize + ",");
              outputLineHalfEpoch.Append(hiddenNodes + ",");
              outputLineEpoch.Append(hiddenNodes + ",");
              outputLineHalfEpoch.Append(learningRate + ",");
              outputLineEpoch.Append(learningRate + ",");
              outputLineHalfEpoch.Append(momentum + ",");
              outputLineEpoch.Append(momentum + ",");
              for (int i = 0; i < trainImages.Length; i++)
              {
                //network.printNetwork();
                decimal[] targetVector = Utility.convertDigitToVector(trainLabels[i]);
                network.propogateInputForwards(trainImages[i]);
                network.calcErrorForPredictedValForOutputAndHiddenUnits(targetVector);
                network.updateEachNetworkDeltaWeight(learningRate);
                if ((i + 1) % batchSize == 0) {
                  network.updateEachNetworkWeight(batchSize, momentum);
                  // network.resetEachNetworkDeltaWeight();
                }
                if (i == (trainImages.Length / 2 - 1))
                {
                  outputLineHalfEpoch.Append(epoch + ",");
                  Console.WriteLine("At half epoch #" + epoch + " using testing data");
                  outputLineHalfEpoch.Append(predictOutputAndComputeErrors(network.neurons, testImages, testLabels) + ",");
                  //Console.WriteLine("At half epoch #" + epoch + " using training data");
                  //outputLineHalfEpoch.Append(predictOutputAndComputeErrors(network.neurons, trainImages, trainLabels));
                  output.Add(outputLineHalfEpoch.ToString());
                }
              }
              outputLineEpoch.Append(epoch + ",");
              Console.WriteLine("At epoch #" + epoch + " using testing data");
              outputLineEpoch.Append(predictOutputAndComputeErrors(network.neurons, testImages, testLabels) + ",");
              //Console.WriteLine("At epoch #" + epoch + " using training data");
              //outputLineEpoch.Append(predictOutputAndComputeErrors(network.neurons, trainImages, trainLabels));
              output.Add(outputLineEpoch.ToString());
            }
          }
        }
      }
      File.WriteAllLines(filePath + "summary_" + batchSize + ".csv", output);
    }
    public static string predictOutputAndComputeErrors(Neuron[][] network, decimal[][] inputDataSet, int[] targetLabels) {
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
      return mean + "," + accuracy;
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
        network.updateEachNetworkWeight(batchSize, 0);
        network.resetEachNetworkDeltaWeight();
      }
    }
  } // Program class ends
}

