using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
  class Program
  {
    public const int USERSCOUNT = 28978;
    public const string filePath = @"C:\Users\nsathya\Desktop\CSE546P-ML\Machine-Learning-Assignments\Assignment 2\Code\Assignment2\bin\Debug\";
    public static Dictionary<int, Dictionary<int, double>> userData = new Dictionary<int, Dictionary<int, double>>();
    public static Dictionary<int, double> meanUserVoteData = new Dictionary<int, double>();
    public static Dictionary<int, Dictionary<int, double>> movieData = new Dictionary<int, Dictionary<int, double>>();
    public static Dictionary<int, int> userMap = new Dictionary<int, int>();
    public static Double[][] weights = new Double[USERSCOUNT][];
    // public static Dictionary<int, Dictionary<int, double>> weights = new Dictionary<int, Dictionary<int, double>>();
    public static string trainingFileName = filePath + "TrainingRatings.txt";
    public static string testingFileName = filePath + "TestingRatings.txt";
    public static string weightFileName = filePath + "weights.txt";
    public static string partialOutputFileName = filePath + "output_Threshold_";
    public static string outputFileExtension = ".txt";
    public static double[] thresholds = new double[] { 0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5 };

    static void Main(string[] args)
    {
      loadTrainingDataIntoDict(trainingFileName);
      loadMeanUserVoteDataIntoDict();
      getWeightForUsers();
      foreach (double threshold in thresholds) {
        evaluateTestData(testingFileName, threshold);
      }
    }

    public static void loadTrainingDataIntoDict(string fullFileName)
    {
      try
      {
        using (StreamReader sr = new StreamReader(fullFileName))
        {
          int userIdx = 0;
          String line = sr.ReadLine();
          while (line != null)
          {
            String[] data = line.Split(new Char[] { ',' });
            int movieId = Convert.ToInt32(data[0]);
            int userId = Convert.ToInt32(data[1]);
            double rating = Convert.ToDouble(data[2]);

            if (!userData.ContainsKey(userId))
            {
              Dictionary<int, double> movieRatingGivenByUser = new Dictionary<int, double>();
              movieRatingGivenByUser.Add(movieId, rating);
              userData.Add(userId, movieRatingGivenByUser);
            }
            else
            {
              Dictionary<int, double> movieRatings = userData[userId];
              if (!movieRatings.ContainsKey(movieId))
              {
                movieRatings.Add(movieId, rating);
              }
              userData[userId] = movieRatings;
            }

            if (!userMap.ContainsKey(userId))
            {
              userMap.Add(userId, userIdx++);
            }

            if (!movieData.ContainsKey(movieId))
            {
              Dictionary<int, double> userRatingGivenMovie = new Dictionary<int, double>();
              userRatingGivenMovie.Add(userId, rating);
              movieData.Add(movieId, userRatingGivenMovie);
            }
            else
            {
              Dictionary<int, double> userGivenRatings = movieData[movieId];
              if (!userGivenRatings.ContainsKey(userId))
              {
                userGivenRatings.Add(userId, rating);
              }
              movieData[movieId] = userGivenRatings;
            }

            line = sr.ReadLine();
          }
        }
      }
      catch (Exception ex)
      {
        Console.WriteLine("Cannot read text file " + fullFileName + " because of exception " + ex.ToString());
      }
    }

    public static void loadMeanUserVoteDataIntoDict()
    {
      foreach (var item in userData)
      {
        Dictionary<int, double> movieRatings = userData[item.Key];
        double sumOfVotes = movieRatings.Sum(rating => rating.Value);
        int countMoviesRated = movieRatings.Count();
        double avgVoteGivenUser = sumOfVotes / countMoviesRated;
        if (!meanUserVoteData.ContainsKey(item.Key))
        {
          meanUserVoteData.Add(item.Key, avgVoteGivenUser);
        }
      }
    }

    public static void getWeightForUsers()
    {
      if (!File.Exists(weightFileName))
      {
        using (StreamWriter file = new StreamWriter(weightFileName))
        {
          int a = 0;
          foreach (var userA in userData)
          {
            var userIdA = userA.Key;
            var moviesOfUserA = userA.Value;
            foreach (var userI in userData)
            {
              var userIdI = userI.Key;
              if (userIdA < userIdI)
              {
                var moviesOfUserI = userI.Value;
                double weight = calcWeight(userIdA, userIdI, moviesOfUserA, moviesOfUserI);
                file.WriteLine("{0},{1},{2}", userIdA, userIdI, weight);
                addToWeightsArray(userIdA, userIdI, weight);
              }
            }
            a++;
            if (a % 1000 == 0)
            {
              Console.WriteLine("coefficient for 1000 users computed");
            }
          }
        }
      }
      else
      {
        using (StreamReader sr = new StreamReader(weightFileName))
        {
          String line = sr.ReadLine();
          while (line != null)
          {
            String[] data = line.Split(new Char[] { ',' });
            int userId1 = Convert.ToInt32(data[0]);
            int userId2 = Convert.ToInt32(data[1]);
            double weight = Convert.ToDouble(data[2]);

            addToWeightsArray(userId1, userId2, weight);

            line = sr.ReadLine();
          }
        }
      }
    }

    public static double calcWeight(int userIdA, int userIdI, Dictionary<int, double> movieOfUserADict, Dictionary<int, double> movieOfUserIDict)
    {
      var meanVoteUserA = meanUserVoteData[userIdA];
      var meanVoteUserI = meanUserVoteData[userIdI];

      double numerator = 0;
      double denomA = 0;
      double denomI = 0;
      foreach (var movie in movieOfUserADict)
      {
        if (!movieOfUserIDict.ContainsKey(movie.Key))
        {
          continue;
        }
        double voteByUserA = movieOfUserADict[movie.Key];
        double voteByUserI = movieOfUserIDict[movie.Key];

        double diffFromMeanUserA = voteByUserA - meanVoteUserA;
        double diffFromMeanUserI = voteByUserI - meanVoteUserI;

        numerator += (diffFromMeanUserA * diffFromMeanUserI);
        denomA += (diffFromMeanUserA * diffFromMeanUserA);
        denomI += (diffFromMeanUserI * diffFromMeanUserI);
      }

      double weight = numerator / (Math.Sqrt(denomA * denomI));
      //if (weight > 1.0 || weight < -1.0)
      //{
      //  Console.WriteLine("Value out of range:" + weight + "," + userIdA + "," + userIdI);
      //}
      return weight;
    }

    public static void addToWeightsArray(int userId1, int userId2, double weight)
    {
      int userIdx1 = userMap[userId1];
      int userIdx2 = userMap[userId2];

      if (weights[userIdx1] == null) {
        weights[userIdx1] = new Double[USERSCOUNT];
      }
      weights[userIdx1][userIdx2] = weight;
    }

    public static void evaluateTestData(string fullFileName, double threshold) {
      try
      {
        double meanAbsError = 0;
        double rmsError = 0;
        int testDataCount = 0;
        using (StreamReader sr = new StreamReader(fullFileName))
        {
          String line = sr.ReadLine();
          while (line != null)
          {
            String[] data = line.Split(new Char[] { ',' });
            int movieId = Convert.ToInt32(data[0]);
            int userId = Convert.ToInt32(data[1]);
            double rating = Convert.ToDouble(data[2]);
            double predictedRating = predictVote(movieId, userId, threshold);
            double diffInRating = predictedRating - rating;
            meanAbsError += Math.Abs(diffInRating);
            rmsError += Math.Pow(diffInRating, 2);
            testDataCount += 1;
            line = sr.ReadLine();
          }
        }
        meanAbsError = meanAbsError / testDataCount;
        rmsError = rmsError / testDataCount;

        string outputFileName = partialOutputFileName + threshold.ToString() + outputFileExtension;
        outputToFile(outputFileName, threshold, meanAbsError, rmsError, testDataCount);

      }
      catch (Exception ex)
      {
        Console.WriteLine("Cannot read text file " + fullFileName + " because of exception " + ex.ToString());
      }
    }

    public static double predictVote(int movieId, int userId, double threshold)
    {
      double rating = 0;
      double alpha = 0;
      double deviation = 0;
      foreach (var userRating in movieData[movieId]) {
        var weight = getWeightForPairOfUsers(userMap[userId], userMap[userRating.Key]);
        if (weight < threshold || Double.IsNaN(weight)) {
          continue;
        }
        alpha += Math.Abs(weight);
        deviation += (weight) * (userRating.Value - meanUserVoteData[userRating.Key]);
      }
      Console.WriteLine("Predicting user {0} movie {1} with deviation {2}, alpha {3}", userId, movieId, deviation, alpha);
      if (alpha != 0d) {
        deviation /= alpha;
      }

      rating = deviation;
      if (meanUserVoteData.ContainsKey(userId))
      {
        rating = meanUserVoteData[userId] + deviation;
      }
      return rating;
    }

    public static double getWeightForPairOfUsers(int userId1, int userId2) {
      try
      {
        if (userId1 < userId2)
        {

          return weights[userId1][userId2];
        }
        if (userId2 < userId1)
        {
          return weights[userId2][userId1];
        }
        return 0;
      }
      catch (Exception ex) {
        Console.WriteLine("Key not found : " + userId1.ToString() + "," + userId2.ToString());
      }
      return 0;
    }

    public static void outputToFile(string fileName, double threshold, double meanAbsError, double rmsError, int testDataCount) {
      using (StreamWriter file = new StreamWriter(fileName)) {
        Console.WriteLine("Threshold : {0}", threshold);
        Console.WriteLine("Mean Absolute Error : {0}", meanAbsError);
        Console.WriteLine("Root Mean Square Error : {0}", rmsError);
        Console.WriteLine("Test Data Count : {0}", testDataCount);
        file.WriteLine("Threshold : {0}", threshold);
        file.WriteLine("Mean Absolute Error : {0}", meanAbsError);
        file.WriteLine("Root Mean Square Error : {0}", rmsError);
        file.WriteLine("Test Data Count : {0}", testDataCount);
      }        
    }
  }
}
