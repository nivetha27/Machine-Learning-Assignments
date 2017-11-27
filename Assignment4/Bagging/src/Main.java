import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

import weka.classifiers.Evaluation;
//import weka.classifiers.trees.J48;
import weka.classifiers.trees.REPTree;
import weka.core.Instance;
import weka.core.Instances;

public class Main {
	public static Random rand = new Random();
	public static String filePath = "C:\\Users\\nsathya\\Desktop\\Assignment4\\";
	public static void main(String[] args) throws Exception {
		Instances trainInstances = loadFromFile(filePath + "diabetes_train.arff");
		Instances testInstances = loadFromFile(filePath + "diabetes_test.arff");
		int n_tests = testInstances.numInstances();
		int maxClasses = 2;
		int bootstrap_samples = 10;
		int[] sampleSizes = new int[] {1,3,5,10,20};
		
		int[][] results = new int[n_tests][];
		for(int i = 0; i < n_tests; i++) {
			results[i] = new int[bootstrap_samples];
		}
		
		int[] depths = new int[] {1,2,3};
		/*for(int depth: depths) {
			int positive = 0;
			REPTree tree = buildTree(trainInstances, depth);
			int predicted[][] = new int[n_tests][];
			for(int i = 0; i < n_tests; i++) {				
				Instance test =  testInstances.instance(i);
				int predictedVal = classify(tree, test);
				int actual = Integer.parseInt(test.classAttribute().value((int)test.classValue()));
				if (predictedVal == actual)
					positive += 1;
				predicted[i] = new int[] {predictedVal};
			}
			System.out.println(">>depth=" + depth + ",accuracy=" + positive * 100.0/n_tests + "(" + positive + "/" + n_tests + ")");
			double[] metrics = biasVar(testInstances, predicted, n_tests, 1, maxClasses);
			System.out.println(",Depth=" + depth + ",Bias=" + metrics[1] + ",Variance=" + metrics[5] + ",Loss=" + metrics[0]);
			// Evaluation evaluation = new Evaluation(trainInstances);
			// evaluation.evaluateModel(tree, testInstances);
			// System.out.println(evaluation.toSummaryString());
		}*/
		for (int sampleSize: sampleSizes) {
			for(int depth: depths) {
				double sumAccuracy = 0;
				for(int j =0; j < bootstrap_samples; j++) {
					REPTree[] treeSets = new REPTree[sampleSize];
					for(int i = 0; i < sampleSize; i++) {
						//prinTopN(trainInstances, 10);
						Instances sampledTrain = trainInstances.resample(rand);
						//prinTopN(sampledTrain, 10);
						treeSets[i] = buildTree(sampledTrain, depth);
					}
					results = bagging(treeSets, testInstances, results, j, maxClasses);
					sumAccuracy += accuracy(testInstances, n_tests, results, j);
				}
				double[] metrics = biasVar(testInstances, results, n_tests, bootstrap_samples, maxClasses);
				System.out.println("#trainSets=" + sampleSize + ",Depth=" + depth + ",Bias=" + metrics[1] + ",Variance=" + metrics[5] + ",Accuracy=" + sumAccuracy/bootstrap_samples + ",Loss=" + metrics[0]);
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
	
	public static REPTree buildTree(Instances train, int depth) throws Exception {
		REPTree tree = new REPTree();
		tree.setNoPruning(true);
		tree.setMaxDepth(depth);
		tree.buildClassifier(train);
		return tree;
	}
	
	public static int[][] bagging(REPTree[] treeSet, Instances testInstances, int[][] results, int idx, int maxClasses) throws Exception {
		int len = testInstances.numInstances();
		for(int i = 0; i < len; i++) {
			Instance test = testInstances.instance(i);
			int numTrees = treeSet.length;
			int[] predsx = new int[numTrees];
			int j = 0;
			for(REPTree tree: treeSet) {
				int predicted = classify(tree, test);
				predsx[j] = predicted;
				++j;
			}
			int classx = Integer.parseInt(test.classAttribute().value((int)test.classValue()));
			double[] metrics = biasvarx(classx, predsx, treeSet.length, maxClasses);
			double majclass = metrics[3];
			results[i][idx] = (int)majclass;
		}
		return results;
	}
	
	// //0 - loss, 1 - bias, 2 - varp, 3 - varn, 4 - varc, 5 - var, 6 - accuracy
	public static double[] biasVar(Instances testInstances, int[][] predicted, int ntestexs, int ntrsets, int maxClasses) {
		double[] metrics = new double[] {0.0,0.0,0.0,0.0,0.0, 0.0,0.0}; //0 - loss, 1 - bias, 2 - varp, 3 - varn, 4 - varc, 5 - var, 6-accuracy
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
	
	// 0 - lossx, 1 - biasx, 2 - varx, 3 - predicted val
	public static double[] 	biasvarx(int classx, int[] predsx, int ntrsets, int maxClasses) {
		double[] metrics = new double[] {0.0,0.0,0.0,0.0}; // 0 - lossx, 1 - biasx, 2 - varx
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
		metrics[3] = majclass;
		return metrics;
	}
	
	public static double accuracy(Instances dataset, int size, int[][] results, int idx) {
		int positive = 0;
		for(int i = 0; i < size; i++) {
			Instance data = dataset.instance(i);
			int classx = Integer.parseInt(data.classAttribute().value((int)data.classValue()));
			if (classx == results[i][idx]) {
				++positive;
			}
		}
		return positive * 100.0/size;
	}
}
