using System;
using System.Text;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using System.Threading.Tasks;
using System.Threading;

namespace Assignment3
{
  class Program
  {
    public static string filePath = @"C:\Users\nivet_000\Desktop\CSE546P\Assignment 3\MNIST_PCA\"; //@"C:\Users\nsathya\Desktop\CSE546P-ML\Machine-Learning-Assignments\Assignment 3\MNIST_PCA\";
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
    public static int numInputNodes = 50;
    public static int numOutputNodes = 10;

    static void Main(string[] args)
    {
      try
      {
        var options = new Options();
        bool isValid = CommandLine.Parser.Default.ParseArgumentsStrict(args, options);
        Console.WriteLine("\nBegin\n");
        int batchSize = 100;
        int numHiddenNodes = 10;
        decimal learningRates = 0.1M;
        decimal momentums = 0; // 0.8M;
        int numLayers = 3;
        int epochs = 100;
        if (isValid) {
          if (options.help) {
            Console.WriteLine(CommandLine.Text.HelpText.AutoBuild(options));
            return;
          }

          if(options.numHiddenNodes != null) {
            numHiddenNodes = (int)options.numHiddenNodes;
          }

          if(options.learningRate != null) {
            learningRates = (decimal)options.learningRate;
          }

          if(options.momentum != null) {
            momentums = (decimal)options.momentum;
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

        //var data = readImagesAndLabels(trainingImagesFile, trainingLabelFile);
        var data = readAndNormalizeInputData(trainingImagesFile, trainingLabelFile);
        trainImages = data.Item1;
        trainLabels = data.Item2;
        //data = readImagesAndLabels(testImagesFile, testLabelFile);
        data = readAndNormalizeInputData(testImagesFile, testLabelFile);
        testImages = data.Item1;
        testLabels = data.Item2;
        runBackpropogationAlgoUsingBatching(numLayers, epochs, numHiddenNodes, learningRates, momentums, batchSize, false, 10);

        //sanityCheckBackPropogationAlgo();
        //eightInputOutputTest();
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

    public static Tuple<decimal[][], int[]> readAndNormalizeInputData(string imagesFileName, string labelFileName)
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

      //for (int col = 0; col < pcaImages[0].Length; col++)
      //{
      //  decimal max = Int32.MinValue, min = Int32.MaxValue;
      //  for (int row = 0; row < pcaImages.Length; row++)
      //  {
      //    if (pcaImages[row][col] > max)
      //    {
      //      max = pcaImages[row][col];
      //    }
      //    if (pcaImages[row][col] < min)
      //    {
      //      min = pcaImages[row][col];
      //    }
      //  }
      //  for (int row = 0; row < pcaImages.Length; row++)
      //  {
      //    pcaImages[row][col] = (pcaImages[row][col] - min) / (max - min);
      //  }
      //}

      for (int col = 0; col < pcaImages[0].Length; col++)
      {
        var attr = new List<decimal>();
        for (int row = 0; row < pcaImages.Length; row++)
        {
          attr.Add(pcaImages[row][col]);
        }
        decimal avg = attr.Average();
        decimal stddev = (decimal)Math.Sqrt((attr.Sum(v => Math.Pow((double)(v - avg), 2))) / (attr.Count - 1));
        for (int row = 0; row < pcaImages.Length; row++)
        {
          pcaImages[row][col] = (pcaImages[row][col] - avg) / stddev;
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
    public static void runBackpropogationAlgoUsingBatching(int numLayers, int epochs, int numHiddenNodes, decimal learningRate, decimal momentum, int batchSize = 1, bool isRelu =false, int numBits = 10)
    {
      Console.WriteLine("Hidden node count : " + numHiddenNodes + ", Learning rate : " + learningRate + ", Momentum : " + momentum + ", Batch Size : " + batchSize);
      var network = new NeuralNetwork(numLayers, numOutputNodes, numInputNodes, numHiddenNodes);
      network.createFeedForwardNetwork();
      //network.testCreateFeedForwardNetwork();
      network.printNetwork(filePath + "startNetwork.txt");
      int epoch = 1;
      using (StreamWriter sw = new StreamWriter(filePath + "Release_InputNormaliztionSummary_lr_" + learningRate + "_m_" + momentum + "_" + batchSize + ".csv"))
      {
        //sw.WriteLine("Batch Size, Hidden Nodes Count, Learning Rate, Momentum, Epoch, Test Square Error, Test Accuracy, Train Square Error, Train Accuracy");
        sw.WriteLine("Batch Size, Hidden Nodes Count, Learning Rate, Momentum, Epoch, Test Square Error, Test Accuracy");
        Stopwatch stopWatch = new Stopwatch();
        for (; epoch <= epochs; epoch++)
        {
          stopWatch.Start();
          for (int i = 0; i < trainImages.Length; i++)
          {
            //network.printNetwork();
            decimal[] targetVector = Utility.convertDigitToVector(trainLabels[i], numBits);
            if (isRelu)
            {
              network.propogateReluInputForwards(trainImages[i]);
              network.calcReluErrorForPredictedValForOutputAndHiddenUnits(targetVector);
            }
            else
            {
              network.propogateInputForwards(trainImages[i]);
              network.calcErrorForPredictedValForOutputAndHiddenUnits(targetVector);
            }
            network.updateEachNetworkDeltaWeight(learningRate);
            if ((i + 1) % batchSize == 0)
            {
              network.updateEachNetworkWeight(batchSize, momentum);
            }
            if (i == (trainImages.Length / 2 - 1))
            {
              //network.printNetwork(filePath + "endNetwork.txt");
              Console.WriteLine("At half epoch #" + epoch + " using testing data");
              Tuple<decimal, double> statsHE = predictOutputAndComputeErrors(network.neurons, testImages, testLabels);
              //Console.WriteLine("At half epoch #" + epoch + " using training data");
              //Tuple<decimal, double> statsTrainHE = (predictOutputAndComputeErrors(network.neurons, trainImages, trainLabels));
              sw.WriteLine(String.Format("{0},{1},{2},{3},{6},{4},{5}", batchSize, numHiddenNodes, learningRate, momentum, statsHE.Item1, statsHE.Item2, epoch));
            }
          }
          Console.WriteLine("At epoch #" + epoch + " using testing data");
          Tuple<decimal, double> stats = predictOutputAndComputeErrors(network.neurons, testImages, testLabels);
          //Console.WriteLine("At epoch #" + epoch + " using training data");
          //Tuple<decimal, double> statsTrain = (predictOutputAndComputeErrors(network.neurons, trainImages, trainLabels));
          sw.WriteLine(String.Format("{0},{1},{2},{3},{6},{4},{5}", batchSize, numHiddenNodes, learningRate, momentum, stats.Item1, stats.Item2, epoch));
          stopWatch.Stop();
          Console.WriteLine(stopWatch.Elapsed.ToString());
          stopWatch.Reset();
        }
        network.printNetwork(filePath + "endNetwork.txt");
      }
    }
    public static Tuple<decimal, double> predictOutputAndComputeErrors(Neuron[][] network, decimal[][] inputDataSet, int[] targetLabels, int numBits = 10) {
      var inputNetwork = new NeuralNetwork(network);
      int i = 0;
      int positive = 0;
      decimal mean = 0;
      int predictedDigit;
      foreach (decimal[] data in inputDataSet)
      {
        inputNetwork.propogateInputForwards(data);
        predictedDigit = inputNetwork.predictedDigit();
        if (predictedDigit == targetLabels[i])
        {
          ++positive;
        }
        mean += inputNetwork.computeSquaredError(Utility.convertDigitToVector(targetLabels[i], numBits));
        ++i;
      }
      mean = mean / (2 * inputDataSet.Length);
      double accuracy = positive * 100.0/ inputDataSet.Length;
      Console.WriteLine("Mean Square Error : " + mean + ", Accuracy : " + accuracy);
      return new Tuple<decimal, double>(mean, accuracy);
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
      }
    }

    public static void eightInputOutputTest() {
      numInputNodes = numOutputNodes = 8;
      trainImages = new decimal[8][];
      trainLabels = new int[8];
      for (int i = 0; i < 8; i++)
      {
        trainImages[i] = new decimal[8];
        trainLabels[i] = i;
        for (int j = 0; j < 8; j++)
        {
          if (j == i)
          {
            trainImages[i][j] = 1;
          }
          else
          {
            trainImages[i][j] = 0;
          }
        }
      }
      testImages = trainImages;
      testLabels = trainLabels;
      runBackpropogationAlgoUsingBatching(3, 100, 3, 0.3M, 0, 1, false, 8);
    }
  } // Program class ends
}

