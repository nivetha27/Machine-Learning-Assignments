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
        public const string filePath = @"C:\Users\nivet_000\Downloads\netflix_data\";
        public static Dictionary<int, Dictionary<int, double>> userData = new Dictionary<int, Dictionary<int, double>>();
        public static Dictionary<int, double> meanUserVoteData = new Dictionary<int, double>();
        public static Dictionary<int, Dictionary<int, double>> movieData = new Dictionary<int, Dictionary<int, double>>();
        public static Dictionary<int, int> userMap = new Dictionary<int, int>();
        public static Double[][] weights = new Double[USERSCOUNT][];
        public static string trainingFileName = filePath + "TrainingRatings.txt";
        public static string testingFileName = filePath + "TestingRatings.txt";
        public static string weightFileName = filePath + "weights.txt";

        static void Main(string[] args)
        {

            loadTrainingDataIntoDict(trainingFileName);
            loadMeanUserVoteDataIntoDict();
            getWeightForUsers();
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

                        if (!userMap.ContainsKey(userId)) {
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
                    for (int a = 0; a < userData.Count(); a++)
                    {
                        var userIdA = userData.Keys.ElementAt(a);
                        var moviesOfUserA = userData.Values.ElementAt(a);
                        for (int i = a + 1; i < userData.Count(); i++)
                        {
                            var userIdI = userData.Keys.ElementAt(i);
                            // if (weights[a][i] == -1)
                            {
                                var moviesOfUserI = userData.Values.ElementAt(i);
                                double weight = calcWeight(userIdA, userIdI, moviesOfUserA, moviesOfUserI);
                                file.WriteLine("{0},{1},{2}", userIdA, userIdI, weight);
                                addToWeightsArray(userIdA, userIdI, weight);
                            }
                        }

                        if (a % 1000 == 0) {
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

            var moviesInCommon = movieOfUserADict.Keys.Intersect(movieOfUserIDict.Keys)
                                    .ToDictionary(t => t, t => movieOfUserADict[t]);
            double numerator = 0;
            double denomA = 0;
            double denomI = 0;
            foreach (var movie in moviesInCommon)
            {
                double voteByUserA = movieOfUserADict[movie.Key];
                double voteByUserI = movieOfUserIDict[movie.Key];

                double diffFromMeanUserA = voteByUserA - meanVoteUserA;
                double diffFromMeanUserI = voteByUserI - meanVoteUserI;

                numerator += (diffFromMeanUserA * diffFromMeanUserI);
                denomA += (diffFromMeanUserA * diffFromMeanUserA);
                denomI += (diffFromMeanUserI * diffFromMeanUserI);
            }

            double weight = numerator / (Math.Sqrt(denomA * denomI));
            if (weight > 1 || weight < -1)
            {
                Console.WriteLine("Value out of range:" + weight + "," + userIdA + "," + userIdI);
            }
            return weight;
        }

        /**
         * Assumes userId1 is less than userId2
         * **/
        public static void addToWeightsArray(int userId1, int userId2, double weight) {
            int userIdx1 = userMap[userId1];
            int userIdx2 = userMap[userId2];

            if (weights[userIdx1] == null) {
                weights[userIdx1] = new double[USERSCOUNT];
            }            
            weights[userIdx1][userIdx2] = weight;
        }
    }
}
