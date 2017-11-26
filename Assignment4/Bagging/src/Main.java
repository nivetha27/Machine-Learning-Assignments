import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;
import weka.filters.Filter;
//import weka.classifiers.trees.J48;
import weka.filters.supervised.instance.Resample;
import weka.classifiers.trees.REPTree;
import weka.core.Instance;
import weka.core.Instances;

public class Main {
	public static Random rand = new Random();
	public static String filePath = "C:\\Users\\nsathya\\Desktop\\Assignment4\\";
	public static void main(String[] args) throws Exception {
		Instances trainInstances = loadFromFile(filePath + "train.arff");
		Instances testInstances = loadFromFile(filePath + "test.arff");
		boolean shouldSample = true;
		int[] sampleSizes = new int[] {1,3,5,10,20};
		int[] depths = new int[] {1,2,3};
		for (int sampleSize: sampleSizes) {
			for(int depth: depths) {
				REPTree[] treeSet = new REPTree[sampleSize];
				for(int i = 0; i < sampleSize; i++) {
					int n = rand.nextInt(sampleSize);
					//prinTopN(trainInstances, 10);
					Instances sampledTrain = sampleData(trainInstances, n, shouldSample);
					//prinTopN(sampledTrain, 10);
					treeSet[i] = buildTree(sampledTrain, depth);
				}
				int[][] predicted = bagging(treeSet, testInstances);
				double[] metrics = biasVar(testInstances, predicted, testInstances.numInstances(), sampleSize);
				System.out.println("#trainSets=" + sampleSize + ",Depth=" + depth + ",Bias=" + metrics[1] + ",Variance=" + metrics[5] + ",Accuracy=" + metrics[6] + ",Loss=" + metrics[0]);
			}
		}
	}
	
	public static int classify(REPTree tree, Instance test) throws Exception {
		int predicted = (int)tree.classifyInstance(test);
		return predicted;
	}
	
	public static void prinTopN(Instances data, int n) {
		System.out.println("\n");
		for(int i = 0; i < n; i++) {
			System.out.println(data.instance(i));
		}
		System.out.println("\n");
	}
	
	public static Instances loadFromFile(String fullFileName) throws Exception{
		BufferedReader reader = new BufferedReader(new FileReader(fullFileName));
		Instances data = new Instances(reader);
		reader.close();
		// setting class attribute
		data.setClassIndex(data.numAttributes() - 1);
		return data;
   }
	
	public static Instances sampleData(Instances dataset, int seed, boolean shouldSample) throws Exception {
		if (!shouldSample)
			return dataset;
		Resample filter = new Resample();
		filter.setBiasToUniformClass(1);
		filter.setRandomSeed(seed);
		filter.setSampleSizePercent(100.0);
		filter.setInputFormat(dataset);
		Instances sampledDataset = Filter.useFilter(dataset, filter);
		return sampledDataset;
	}
	
	public static REPTree buildTree(Instances train, int depth) throws Exception {
		REPTree tree = new REPTree();
		tree.setNoPruning(true);
		tree.setMaxDepth(depth);
		tree.buildClassifier(train);
		return tree;
	}
	
	public static int[][] bagging(REPTree[] treeSet, Instances testInstances) throws Exception {
		int len = testInstances.numInstances();
		int[][] predictedVals = new int[len][];
		for(int i = 0; i < len; i++) {
			Instance test = testInstances.instance(i);
			int numTrees = treeSet.length;
			predictedVals[i] = new int[numTrees];
			int j =0;
			for(REPTree tree: treeSet) {
				int predicted = classify(tree, test);
				predictedVals[i][j] = predicted;
				++j;
			}
		}
		return predictedVals;
	}
	
	// //0 - loss, 1 - bias, 2 - varp, 3 - varn, 4 - varc, 5 - var, 6 - accuracy
	public static double[] biasVar(Instances testInstances, int[][] predicted, int ntestexs, int ntrsets) {
		double[] metrics = new double[] {0.0,0.0,0.0,0.0,0.0, 0.0,0.0}; //0 - loss, 1 - bias, 2 - varp, 3 - varn, 4 - varc, 5 - var, 6-accuracy
		int maxClasses = 2;
		for (int e = 0; e < ntestexs; e++) {
			double lossx = 0, biasx = 0, varx = 0;
			Instance data = testInstances.instance(e);
			int classx = Integer.parseInt(data.classAttribute().value((int)data.classValue()));
			int[] predsx = predicted[e];			
			double[] metricsx =	biasvarx(classx, predsx, ntrsets, maxClasses);
			lossx = metricsx[0];
			biasx = metricsx[1];
			varx = metrics[2];
			metrics[0] += lossx;
			metrics[1] += biasx;
			if(biasx != 0.0) {
				metrics[3] += varx;
				metrics[4] += 1.0;
				metrics[4] -= lossx;
			} else {
				metrics[2] += varx;
				metrics[6] += 1.0;
			}
		}
		metrics[0] /= ntestexs;
		metrics[1] /= ntestexs;
	   	metrics[5] = metrics[0] - metrics[1];
	   	metrics[2] /= ntestexs;
	   	metrics[3] /= ntestexs;
	   	metrics[4] /= ntestexs;
	   	metrics[6] *= 100.0/ntestexs;
	   	return metrics;
	}
	
	// 0 - lossx, 1 - biasx, 2 - varx
	public static double[] 	biasvarx(int classx, int[] predsx, int ntrsets, int maxClasses) {
		double[] metrics = new double[] {0.0,0.0,0.0}; // 0 - lossx, 1 - biasx, 2 - varx
		int c, t, majclass = -1, nmax = 0;
		int[] nclass = new int[maxClasses];
		for (c = 0; c < maxClasses; c++) {
		   nclass[c] = 0;
		}
		for (t = 0; t < ntrsets; t++) {
		  nclass[predsx[t]]++;
		}
		for (c = 0; c < maxClasses; c++) {
		  if (nclass[c] > nmax) {
		    majclass = c;
		    nmax = nclass[c];
		  }
		}
		metrics[0] = 1.0 - nclass[classx] * 1.0/ntrsets;
		metrics[1] = 0.0;
		if (majclass != classx) {
		 metrics[1] = 1.0;
		}
		metrics[2] = 1.0 - nclass[majclass] * 1.0/ ntrsets;
		return metrics;
	}
}
