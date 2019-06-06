import csv
import sys
import operator
import matplotlib.pyplot as plt
import numpy as np

maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

# def getData():
# 	#dictionary of keywords and number of occurenes of each
# 	keyword_count = {}
# 	tagnum = {}
# 	ifile = open('joined_data.csv', "rb")
# 	reader = csv.reader(ifile)
# 	reader = list(reader)
# 	num_post = 0
# 	data_count = 0
# 	for row in reader[1:]:
# 		num_post += 1
# 		keywords_list = row[5].split(" ")
# 		num = len(keywords_list)
# 		for i in keywords_list:
# 			data_count+=1
# 			if i in keyword_count:
# 				keyword_count[i]+=1
# 			else:
# 				keyword_count[i]=1
# 	ifile.close()

def getData():
	#dictionary of keywords and number of occurenes of each
	keywords = ['javascript', 'java', 'c#', 'php', 'android', 'jquery', 'python', 'html', 'c++', 'ios', 'css', 'mysql', 'sql', 'asp.net', 'objective-c', 'ruby-on-rails', '.net', 'c', 'iphone', 'arrays']
	keyword_count = {'javascript':{}, 'java':{}, 'c#':{}, 'php':{}, 'android':{}, 'jquery':{}, 'python':{}, 'html':{}, 'c++':{}, 'ios':{}, 'css':{}, 'mysql':{}, 'sql':{}, 'asp.net':{}, 'objective-c':{}, 'ruby-on-rails':{}, '.net':{}, 'c':{}, 'iphone':{}, 'arrays':{}}
	ifile = open('joined_data.csv', "rb")
	reader = csv.reader(ifile)
	reader = list(reader)
	num_post = 0
	data_count = 0
	for row in reader[1:]:
		num_post += 1
		keywords_list = row[5].split(" ")
		num = len(keywords_list)
		for i in keywords_list:
			for j in keywords_list:
				if i == j:
					continue
				if i in keywords and j in keywords:
					if j in keyword_count[i]:
						keyword_count[i][j]+=1
					else:
						keyword_count[i][j] = 1
					if i in keyword_count[j]:
						keyword_count[j][i]+=1
					else:
						keyword_count[j][i] = 1
			data_count+=1
	ifile.close()

	print keyword_count


	# label = [1,2,3,4,5]
	# percentage_occur = [100*132393/float(num_post),100*288017/float(num_post),100*320804/float(num_post),100*220435/float(num_post),100*140919/float(num_post)]
	# index = np.arange(len(label))
	# plt.bar(index, percentage_occur)
	# plt.xlabel('Number of Tags', fontsize=10)
	# plt.ylabel('Percentage of Posts with Specific Number of Tags', fontsize=10)
	# plt.xticks(index, label, fontsize=10)
	# plt.title('Percentage of Posts with 1,2,3,4,5 Tags')
	# plt.show()


	keyword_count = sorted(keyword_count.items(),key=operator.itemgetter(1),reverse = True)

	#get number of tags with frequency of 1
	# countone = 0
	# for i in keyword_count:
	# 	if i[1] == 1:
	# 		countone+=1
	# 	else:
	# 		break
	# print countone
	# #number of keywords
	# keyword_num = len(keyword_count)

	# sum_10 = 0

	# for i in keyword_count[0:10]:
	# 	sum_10 += i[1]

	# sum_50 = sum_10

	# for i in keyword_count[10:50]:
	# 	sum_50 += i[1]

	# sum_100 = sum_50

	# for i in keyword_count[50:100]:
	# 	sum_100 += i[1]

	# sum_200 = sum_100

	# for i in keyword_count[100:200]:
	# 	sum_200 += i[1]

	# sum_500 = sum_200

	# for i in keyword_count[200:500]:
	# 	sum_500 += i[1]

	# sum_1000 = sum_500

	# for i in keyword_count[500:1000]:
	# 	sum_1000 += i[1]

		
	# print ("total number of posts:",num_post)
	# print ("total occurences of keywords:",data_count)
	# print ("total number of keywords:",keyword_num)
	# print ("total for top 10 keywords:",100*sum_10/float(data_count))
	# print ("total for top 50 keywords:",100*sum_50/float(data_count))
	# print ("total for top 100 keywords:",100*sum_100/float(data_count))
	# print ("total for top 200 keywords:",100*sum_200/float(data_count))
	# print ("total for top 500 keywords:",100*sum_500/float(data_count))
	# print ("total for top 1000 keywords:",100*sum_1000/float(data_count))

	# label = ['Top 10','Top 50','Top 100','Top 200','Top 500','Top 1000']
	# percentage_occur = [100*sum_10/float(data_count),100*sum_50/float(data_count),100*sum_100/float(data_count),100*sum_200/float(data_count),100*sum_500/float(data_count),100*sum_1000/float(data_count)]

	# # top 10 to 1000 tags and their percentage wrt to all tags
	# index = np.arange(len(label))
	# plt.bar(index, percentage_occur)
	# plt.xlabel('Most Frequently Occurring Tags', fontsize=10)
	# plt.ylabel('Percentage of All Tags', fontsize=10)
	# plt.xticks(index, label, fontsize=10, rotation=30)
	# plt.title('Top tags Constitution of All Tags')
	# plt.show()

	# top 10 tags and their frequency
	label = [keyword_count[i][0] for i in range(0,20)]
	percentage_occur = [keyword_count[i][1] for i in range(0,10)]
	print label
	# index = np.arange(len(label))
	# plt.bar(index, percentage_occur)
	# plt.xlabel('Top 10 Occurred Tags', fontsize=10)
	# plt.ylabel('No. of Occurence', fontsize=10)
	# plt.xticks(index, label, fontsize=10, rotation=30)
	# plt.title('Frequency of Top 10 Tags')
	# plt.show()

				

	return 1



def main():
	getData()

if __name__ == '__main__':
    main()
