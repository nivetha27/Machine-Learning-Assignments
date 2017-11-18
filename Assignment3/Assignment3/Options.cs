using CommandLine;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Assignment3
{
    class Options
    {
      
      [Option('l', "learningRate", HelpText = "Input learning rate.", Required = false)]
      public decimal? learningRate { get; set; }

      [Option('m', "momentum", HelpText = "Input momentum", Required = false)]
      public decimal? momentum { get; set; }

      [Option('n', "hiddenNodes", HelpText = "Input number of hidden nodes", Required = false)]
      public int? numHiddenNodes { get; set; }

      [Option('b', "batchSize", HelpText = "Input batch size", Required = false)]
      public int? batchSize { get; set; }

      [Option('f', "filePath", HelpText = "Input file path to be used", Required = false)]
      public string filePath { get; set; }

      [Option('e', "epochs", HelpText = "Input number of epochs", Required = false)]
      public int? epochs { get; set; }

      [Option('h', "help",  DefaultValue = false, HelpText = "Print this help", Required = false)]
      public bool help { get; set; }
    }
}
