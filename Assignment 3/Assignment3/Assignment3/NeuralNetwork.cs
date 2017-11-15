﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Assignment3
{
  public class NeuralNetwork
  {
    public static decimal clippingValue = 650;
    public int numLayers;
    // 0 index represent number of input nodes, (n-1) index represents number of output node.
    public int[] numNodesInEachLayer;
    public Neuron[][] neurons;
    public NeuralNetwork(int numLayers, int numOutputNodes, int numInputNodes, int numHiddenNodes)
    {
      this.numLayers = numLayers;
      this.numNodesInEachLayer = new int[numLayers];
      this.numNodesInEachLayer[0] = numInputNodes;
      this.numNodesInEachLayer[numLayers - 1] = numOutputNodes;
      for (int i = 1; i < numLayers - 1; i++)
      {
        this.numNodesInEachLayer[i] = numHiddenNodes;
      }
    }

    public NeuralNetwork(Neuron[][] neurons)
    {
      this.neurons = neurons;
    }

    public decimal computeSquaredError(decimal[] expectedOutput)
    {
        decimal sum = 0;
        for (int i = 0; i < this.neurons[this.neurons.Length - 1].Length; i++)
        {
            decimal diff = this.neurons[this.neurons.Length - 1][i].value - expectedOutput[i];
            sum += (diff * diff);
        }
        // Console.WriteLine("Squared Error Sum" + sum);
        return sum;
    }

    public int predictedDigit()
    {
        decimal max = this.neurons[this.neurons.Length - 1][0].value;
        int maxPos = 0;
        for (int i = 1; i < this.neurons[this.neurons.Length - 1].Length; i++)
        {
            if (max < this.neurons[this.neurons.Length - 1][i].value)
            {
                max = this.neurons[this.neurons.Length - 1][i].value;
                maxPos = i;
            }
        }
        // Console.WriteLine("Squared Error Sum" + sum);
        return maxPos;
    }

    public bool isOutputLayerInNetwork(int layerNum)
    {
        return layerNum == (this.numLayers - 1);
    }
    
    public void createFeedForwardNetwork() {
      this.neurons = new Neuron[this.numLayers][];

      int outputLayer = this.numLayers - 1;
      this.neurons[outputLayer] = new Neuron[this.numNodesInEachLayer[outputLayer]];
      for (int i = 0; i < this.numNodesInEachLayer[outputLayer]; i++) {
        this.neurons[outputLayer][i] = new Neuron();
      }

      for (int i = 0; i < this.neurons.Length - 1; i++)
      {
        int numNodesInCurLayer = this.numNodesInEachLayer[i];
        int numNodesInNextLater = this.numNodesInEachLayer[i + 1];
        numNodesInCurLayer += 1; // adding bias of 1 to current layer except for output layer
        this.neurons[i] = new Neuron[numNodesInCurLayer];

        // getting weight for bias first
        this.neurons[i][numNodesInCurLayer - 1] = new Neuron();
        this.neurons[i][numNodesInCurLayer - 1].value = 1;
        this.neurons[i][numNodesInCurLayer - 1].weights = new decimal[numNodesInNextLater];
        this.neurons[i][numNodesInCurLayer - 1].delta = new decimal[numNodesInNextLater];
        //decimal weight = Utility.generateRandomDecimalVal();
        for (int j = 0; j < numNodesInNextLater; j++)
        {
            this.neurons[i][numNodesInCurLayer - 1].weights[j] = Utility.generateRandomDecimalVal(); //weight;
        }

        //getting weights for the non-bias nodes.
        for (int j = 0; j < numNodesInCurLayer - 1; j++)
        //for (int j = 0; j < numNodesInCurLayer; j++)
        {
          this.neurons[i][j] = new Neuron();
          this.neurons[i][j].weights = new decimal[numNodesInNextLater];
          this.neurons[i][j].delta = new decimal[numNodesInNextLater];
          for (int k = 0; k < numNodesInNextLater; k++) {
            this.neurons[i][j].weights[k] = Utility.generateRandomDecimalVal();
          }
        }
      }
    }

    public void propogateInputForwards(decimal[] imagePCAComponents)
    {
      // initialize input nodes with given imagePCAComponents excluding the bias.
      for (int i = 0; i < this.neurons[0].Length - 1; i++)
      //for (int i = 0; i < this.neurons[0].Length; i++)
      {
        this.neurons[0][i].value = imagePCAComponents[i];
      }

      // starting from the hidden layer to the output layer
      for (int i = 1; i < this.neurons.Length; i++) {
        int numNeuronsInCurLayer = this.neurons[i].Length;
        // comment start
        if (!this.isOutputLayerInNetwork(i))
        {
           numNeuronsInCurLayer -= 1; // exculding the bias here as the value is already set to 1 for it.
        }// comment end
        for (int j = 0; j < numNeuronsInCurLayer; j++) {
          this.neurons[i][j].value = 0;
          for (int k = 0; k < this.neurons[i - 1].Length; k++) {
            this.neurons[i][j].value += this.neurons[i - 1][k].value * this.neurons[i - 1][k].weights[j];
          }
          this.neurons[i][j].value = Utility.computeSigmoidValue(this.neurons[i][j].value,clippingValue);
        }
      }
    }

    public decimal calcErrorForOutputUnit(decimal predictedOuput, decimal actualOuput ) {
      if (predictedOuput == 0 || predictedOuput == 1)
        return 0;

      decimal error = predictedOuput * (1 - predictedOuput) * (actualOuput - predictedOuput);
      return error;
    }

    public decimal calcErrorForHiddenUnit(Neuron hiddenNode, Neuron[] outputNodes)
    {
      if (hiddenNode.value == 0 || hiddenNode.value == 1)
        return 0;

      decimal sum = 0;
      for (int i = 0; i < hiddenNode.weights.Length; i++)
      {
        sum += (hiddenNode.weights[i] * outputNodes[i].error);
      }
      sum *= hiddenNode.value * (1 - hiddenNode.value);
      return sum;
    }

    public void calcErrorForPredictedValForOutputAndHiddenUnits(decimal[] expectedOutputArr) {
      for (int i = this.neurons.Length - 1; i >= 1; i--) {
        int j = 0;
        foreach (Neuron neuron in this.neurons[i]) {
          if (this.isOutputLayerInNetwork(i))
          {
            neuron.error = this.calcErrorForOutputUnit(neuron.value, expectedOutputArr[j]);
            ++j;
          }
          else {
            neuron.error = this.calcErrorForHiddenUnit(neuron, this.neurons[i+1]);
          }
        }
      }
    }

    public void updateEachNetworkDeltaWeight(decimal learningRate)
    {
      for (int i = 0; i < this.neurons.Length - 1; i++)
      {
        for (int j = 0; j < this.neurons[i].Length; j++)
        {
          for (int k = 0; k < this.neurons[i][j].delta.Length; k++)
          {
            decimal delta = learningRate * this.neurons[i + 1][k].error * this.neurons[i][j].value;
            this.neurons[i][j].delta[k] += delta;
          }
        }
      }
    }


    public void resetEachNetworkDeltaWeight()
    {
      for (int i = 0; i < this.neurons.Length - 1; i++)
      {
        for (int j = 0; j < this.neurons[i].Length; j++)
        {
          for (int k = 0; k < this.neurons[i][j].weights.Length; k++)
          {
            this.neurons[i][j].delta[k] = 0;
          }
        }
      }
    }

    public void updateEachNetworkWeight(int batchSize, decimal momentum)
    {
      for (int i = 0; i < this.neurons.Length - 1; i++)
      {
        for (int j = 0; j < this.neurons[i].Length; j++)
        {
          for (int k = 0; k < this.neurons[i][j].weights.Length; k++)
          {
            decimal weightDelta = (this.neurons[i][j].delta[k] / batchSize) + (momentum * this.neurons[i][j].prevDeltaWeight);
            this.neurons[i][j].weights[k] += weightDelta;
            this.neurons[i][j].prevDeltaWeight = weightDelta;
            this.neurons[i][j].delta[k] = 0;
          }
        }
      }
    }

    public void printNetwork()
    {
        for (int i = 0; i < this.neurons.Length; i++) 
        {
            Console.WriteLine("Layer "  + i);
            for(int j = 0; j < this.neurons[i].Length; j++)
            {
                Console.Write($"[value: {this.neurons[i][j].value}, weights: [");
                if (!this.isOutputLayerInNetwork(i)) {
                    for(int k = 0 ; k < this.neurons[i][j].weights.Length; k++)
                    {
                        Console.Write($"{this.neurons[i][j].weights[k]}");
                        if (k < this.neurons[i][j].weights.Length - 1) 
                        {
                            Console.Write(",");
                        }
                    }
                    Console.Write("], delta: [");
                    for(int k = 0 ; k < this.neurons[i][j].delta.Length; k++)
                    {
                        Console.Write($"{this.neurons[i][j].delta[k]}");
                        if (k < this.neurons[i][j].delta.Length - 1) 
                        {
                            Console.Write(",");
                        }
                    }
                }
                Console.Write("]],");
            }
        }
    }
  }
}
