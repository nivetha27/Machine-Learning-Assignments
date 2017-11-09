using System;
using System.IO;

namespace Assignment3
{
  class Program
  {
    public static string filePath = @"C:\Users\nsathya\Desktop\CSE546P-ML\Machine-Learning-Assignments\Assignment 3\MNIST_PCA\";
    public static string trainingLabelFile = filePath + "train-labels.idx1-ubyte";
    public static string trainingImagesFile = filePath + "train-images-pca.idx2-double";
    public static string testLabelFile = filePath + "t10k-labels.idx1-ubyte";
    public static string testImagesFile = filePath + "t10k-images-pca.idx2-double";
    static void Main(string[] args)
    {
      try
      {
        Console.WriteLine("\nBegin\n");
        int magicImages;
        int numImages;
        int numDimensions;
        long[][] pcaImages;
        int magicLabels;
        int numLabels;
        string[] targetLabels;

        FileStream ifsLabels = new FileStream(trainingLabelFile, FileMode.Open); // test labels
        FileStream ifsImages = new FileStream(trainingImagesFile, FileMode.Open); // test images

        BinaryReader brLabels = new BinaryReader(ifsLabels);
        BinaryReader brImages = new BinaryReader(ifsImages);
        while (brImages.BaseStream.Position != brImages.BaseStream.Length)
        {
          magicImages = Utility.convertBytesToInt(brImages.ReadBytes(sizeof(Int32)));
          numImages = Utility.convertBytesToInt(brImages.ReadBytes(sizeof(Int32)));
          numDimensions = Utility.convertBytesToInt(brImages.ReadBytes(sizeof(Int32)));
          pcaImages = new long[numImages][];
          for (int i = 0; i < numImages; i++)
          {
            pcaImages[i] = new long[numDimensions];
            for (int j = 0; j < numDimensions; j++)
            {
              pcaImages[i][j] = Utility.convertBytesToLong(brImages.ReadBytes(sizeof(Int64)));
            }
          }
        }

        while (brLabels.BaseStream.Position != brLabels.BaseStream.Length)
        {
          magicLabels = Utility.convertBytesToInt(brLabels.ReadBytes(sizeof(Int32)));
          numLabels = Utility.convertBytesToInt(brLabels.ReadBytes(sizeof(Int32)));
          targetLabels = new string[numLabels];
          for (int i = 0; i < numLabels; i++)
          {
            byte lbl = brLabels.ReadByte();
            targetLabels[i] = lbl.ToString();
          }
        }

        //byte[][] pixels = new byte[28][];
        //for (int i = 0; i < pixels.Length; ++i)
        //  pixels[i] = new byte[28];

        //// each test image
        //for (int di = 0; di < 10000; ++di)
        //{
        //  for (int i = 0; i < 28; ++i)
        //  {
        //    for (int j = 0; j < 28; ++j)
        //    {
        //      byte b = brImages.ReadByte();
        //      pixels[i][j] = b;
        //    }
        //  }
        //  byte lbl = brLabels.ReadByte();
        //  DigitImage dImage = new DigitImage(pixels, lbl);
        //  Console.WriteLine(dImage.ToString());
        //  Console.ReadLine();
        //} // each image

        ifsImages.Close();
        brImages.Close();
        ifsLabels.Close();
        brLabels.Close();

        Console.WriteLine("\nEnd\n");
      }
      catch (Exception ex)
      {
        Console.WriteLine(ex.Message);
        Console.ReadLine();
      }
    } // Main method ends
  } // Program class ends
}

