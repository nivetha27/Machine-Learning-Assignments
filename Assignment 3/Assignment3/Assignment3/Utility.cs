using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Assignment3
{
  public class Utility
  {
    public static Boolean isSystemLittleEndian = BitConverter.IsLittleEndian;

    public static int convertBytesToInt(byte[] bytesArr)
    {
      // If the system architecture is little-endian (that is, little end first),
      // reverse the byte array.
      if (isSystemLittleEndian)
        Array.Reverse(bytesArr);

      int intValue = BitConverter.ToInt32(bytesArr, 0);
      Console.WriteLine("int: {0}", intValue);

      return intValue;
    }

    public static long convertBytesToLong(byte[] bytesArr)
    {
      // If the system architecture is little-endian (that is, little end first),
      // reverse the byte array.
      if (isSystemLittleEndian)
        Array.Reverse(bytesArr);

      long longValue = BitConverter.ToInt64(bytesArr, 0);
      Console.WriteLine("int64: {0}", longValue);

      return longValue;
    }
  }
}
