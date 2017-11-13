using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Assignment3
{
  public class Neuron
  {
    public decimal value;
    public decimal[] weights;
    public decimal error;
    public decimal[] delta; //stores delta of weights from each iteration.
  }
}
