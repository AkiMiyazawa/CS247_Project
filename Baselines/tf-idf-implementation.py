import csv
from sklearn.feature_extraction.text import TfidfVectorizer
import string
 
 #create a dictionary with key as tags and values as index of texts with the tag
 #create a list of texts
def getData():
	keywords = {}
	texts = []
	ifile = open('data_subset.csv', "rb")
	reader = csv.reader(ifile)
	reader = list(reader)
	rownum = 0
	#make training dataset
	for row in reader[1:]:
		texts.append(row[0] + ' ' + row[2])
		keywords_list = row[1].split(";")
		for i in range(0,len(keywords_list)):
			keywords_list[i] = string.lower(keywords_list[i])

		keywords[rownum] = keywords_list
		rownum+=1

	ifile.close()

	return keywords,texts


def main():
	tp = 0
	fp = 0
	fn = 0
	keywords,texts = getData()
	tfidf = TfidfVectorizer()
	response = tfidf.fit_transform(texts)
	feature_names = tfidf.get_feature_names()
	for i in range(0,22053):
		if i%100 == 0:
			print i
		col2prob = {}
		for col in response.nonzero()[1]:
			col2prob[col] = response[i,col]
		col2prob = sorted(col2prob.items(), key=lambda x: x[1], reverse=True)
		predicted_list = []
		for col in col2prob:
			# predicted_list.append(string.lower(str(feature_names[col[0]])))
			predicted_list.append(string.lower(feature_names[col[0]].encode('utf-8')))
		predicted_list = predicted_list[0:5]
		tp += len(set(keywords[i]) - (set(keywords[i]) - set(predicted_list)))
		fp += len(set(predicted_list)-set(keywords[i]))
		fn += len(set(keywords[i]) - set(predicted_list))
	precision = tp/float(tp+fp)
	recall = tp/float(tp+fn)
	f1 = 2*precision*recall/(precision+recall)

	print("The precision is", precision)
	print("The recall is", recall)
	print("The f1 score is", f1)



if __name__ == '__main__':
    main()


