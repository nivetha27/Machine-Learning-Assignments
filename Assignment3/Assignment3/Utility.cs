using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Assignment3
{
  public class Utility
  {
    public static Boolean isSystemLittleEndian = BitConverter.IsLittleEndian;
    public static Random rnd = new Random();
    public static int convertBytesToInt(byte[] bytesArr)
    {
      // If the system architecture is little-endian (that is, little end first),
      // reverse the byte array.
      if (isSystemLittleEndian)
        Array.Reverse(bytesArr);

      int intValue = BitConverter.ToInt32(bytesArr, 0);
      return intValue;
    }

    public static double convertBytesToDouble(byte[] bytesArr)
    {
      // If the system architecture is little-endian (that is, little end first),
      // reverse the byte array.
      if (isSystemLittleEndian)
        Array.Reverse(bytesArr);

      double doubleValue = BitConverter.ToDouble(bytesArr, 0);
      return doubleValue;
    }

    public static decimal[] convertDigitToVector(int targetDigit, int numBits = 10)
    {
      decimal[] vector = new decimal[numBits];
      for (int i = 0; i < numBits; i++)
      {
        vector[i] = 0;
        if (i == targetDigit)
        {
          vector[i] = 1;
        }
      }
      return vector;
    }
    public static decimal computeSigmoidValue(decimal value, decimal? clippingValue)
    {
      if (clippingValue != null) {
        if (value > clippingValue) {
          value = (decimal)clippingValue;
        } else if (value < (-1 * clippingValue)) {
          value = (decimal)clippingValue * -1;
        }
      }
      decimal sigmoid = Convert.ToDecimal(1 / (1 + Math.Exp(-1.0 * (double)value)));
      return sigmoid;
    }
    
    public static decimal generateRandomDecimalVal(decimal minimum = -0.5M, decimal maximum = 0.5M)
    {
      decimal weight = (decimal)rnd.NextDouble() * (maximum - minimum) + minimum;
      return weight;
    }

    public static decimal generateWeightsByXavierInitialization(decimal numInput, decimal numOutput)
    {
      decimal maximum = 4 * (decimal)Math.Sqrt(6 / (double)(numInput + numOutput));
      decimal minimum = -1 * maximum;
      decimal weight = (decimal)rnd.NextDouble() * (maximum - minimum) + minimum;
      return weight;
      // return 0.5M;
    }
  }
}
