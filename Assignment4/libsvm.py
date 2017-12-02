import random
from subprocess import *

MaxTestExs=10000
MaxTrSets=100
MaxClasses=100

def biasvarx(classx, predsx, ntrsets):
    nclassc=[]
    nmax=0
    majclass=0
    for c in range(MaxTrSets):
        nclassc.append(0)
    for t in range(len(predsx)):
        nclassc[predsx[t]] += 1
    for c in range(MaxTrSets):
        if nclassc[c] > nmax:
            majclass = c
            nmax = nclassc[c]
    loss = 1.0 - nclassc[classx] / ntrsets
    bias = 0
    if (majclass != classx):
        bias = 1
    var = 1.0 - nclassc[majclass] / ntrsets
    return (loss, bias, var)
    
def biasvar(classes, preds, ntestexs, ntrsets):
  loss = 0.0
  bias = 0.0
  varp = 0.0
  varn = 0.0
  varc = 0.0
  positive = 0.0
  for e in range(ntestexs):
    lossx, biasx, varx = biasvarx(classes[e], preds[e], ntrsets)
    loss += lossx
    bias += biasx
    if (biasx != 0.0):
      varn += varx
      varc += 1.0
      varc -= lossx
    else:
      varp += varx
      positive += 1.0
  
  loss /= ntestexs
  bias /= ntestexs
  var = loss - bias
  varp /= ntestexs
  varn /= ntestexs
  varc /= ntestexs
  positive /= ntestexs
  
  return (loss, bias, var, varp, varn, varc, positive)

def create_models(train_file_path, train_file, trsets, kernel):
	lines=[]
	filename=train_file_path + str(kernel) + train_file
	for line in open(filename):
		lines.append(line)
		
	for k in range(trsets):
		newdata = []
		for i in range(len(lines)):
			newdata.append(lines[random.randint(0,len(lines) - 1)]);

		f = open(filename+str(k), 'w')
		for j in range(len(newdata)):
			f.write(newdata[j])

svmscale_exe = r"C:\Users\nsathya\Desktop\Assignment4\libsvm-3.22\libsvm-3.22\windows\svm-scale.exe"
svmtrain_exe = r"C:\Users\nsathya\Desktop\Assignment4\libsvm-3.22\libsvm-3.22\windows\svm-train.exe"
svmpredict_exe = r"C:\Users\nsathya\Desktop\Assignment4\libsvm-3.22\libsvm-3.22\windows\svm-predict.exe"

file_path = "C:/Users/nsathya/Desktop/Assignment4/libsvm-3.22/libsvm-3.22/windows/k"
train_file_name = "/diabetes_libsvmformat_train.txt"
test_file_name = "/diabetes_libsvmformat_test.txt"

trsets = 10
max_kernel = 3

for i in range(max_kernel + 1):
	create_models(file_path,train_file_name,trsets,i)

resultset = []
for x in range(max_kernel + 1):
	for y in range(trsets):
		train_file_full_name = file_path + str(x) + train_file_name + str(y)
		test_file_full_name = file_path + str(x) + test_file_name
		scaled_file = train_file_full_name + ".scale"
		model_file = train_file_full_name + ".model"
		range_file = train_file_full_name + ".range"
		scaled_test_file = test_file_full_name + ".scale"
		predict_test_file = test_file_full_name + ".predict"
		cmd = '{0} -s "{1}" "{2}" > "{3}"'.format(svmscale_exe, range_file, train_file_full_name, scaled_file)
		#print('Scaling training data...')
		Popen(cmd, shell = True, stdout = PIPE).communicate()
		cmd = '{0} -t {3} "{1}" "{2}"'.format(svmtrain_exe,scaled_file,model_file, x)
		#print('Training...')
		Popen(cmd, shell = True, stdout = PIPE).communicate()
		#print('Output model: {0}'.format(model_file))
		cmd = '{0} -r "{1}" "{2}" > "{3}"'.format(svmscale_exe, range_file, test_file_full_name, scaled_test_file)
		#print('Scaling testing data...')
		Popen(cmd, shell = True, stdout = PIPE).communicate()	

		cmd = '{0} "{1}" "{2}" "{3}"'.format(svmpredict_exe, scaled_test_file, model_file, predict_test_file)
		#print('Testing...')
		Popen(cmd, shell = True).communicate()	

		#print('Output prediction: {0}'.format(predict_test_file))
		predictions=[]
		for line in open(predict_test_file):
			if line != "":
				a = int(line.split()[0])
				if a < 0 :
					a = 0
				predictions.append(a)
		
		resultset.append(predictions)

	originals=[]
	n_tests=0
	for line in open(test_file_full_name):
		n_tests += 1
		if line != "":
			b = int(line.split()[0])
			if b < 0 :
				b = 0
			originals.append(b)
	loss, bias, var, varp, varn, varc, accuracy = biasvar(originals, list(zip(*resultset)), n_tests, trsets)
	print (str(bias) + " " + str(loss) + " " + str(var) + " " + str(varp) + " " + str(varn) + " " + str(varc) + " " + str(accuracy * 100.0))
	resultset = []