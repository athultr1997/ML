import random
import numpy as np

def is_positive_review(review):
	if review[0]>="7":
		return True
	else:
		return False

class Node:
	def _init_(self,data):
		self.left = None
		self.right = None
		self.data = data

class Decision_Tree:
	def __init__(self,total_features):
		self.Nodes = []
		self.features_to_use = range(0,total_features)

	def find_entropy(self,data):
		if len(data)==0:
			return 0
		else:
			p_plus_mag = p_neg_mag = 0
			
			for d in data:
				if is_positive_review(d):
					p_plus_mag += 1
				else:
					p_neg_mag +=1

			p_plus=(float(p_plus_mag)/len(data))
			p_neg=(float(p_neg_mag)/len(data))

			if p_plus == 0:
				entropy = -p_neg*np.log2(p_neg)
			elif p_neg == 0:
				entropy = -p_plus*np.log2(p_plus)  
			else:
				entropy = -p_plus*np.log2(p_plus)-p_neg*np.log2(p_neg) 

			print "p_plus=",p_plus,"p_neg=",p_neg,"entropy=",entropy,"\n"

			return entropy

	def find_IG(self,feature,data):
		entropy_s = self.find_entropy(data)
		data_with_feature = []
		data_without_feature = []
		string = str(feature)+":"
		for d in data: 
			if string in d:
				data_with_feature.append(d)
			else:
				data_without_feature.append(d)

		entropy_s_1 = self.find_entropy(data_with_feature)
		entropy_s_2 = self.find_entropy(data_without_feature)

		print "entropy_s=",entropy_s,"entropy_s_1=",entropy_s_1,"entropy_s_2=",entropy_s_2,"len(data_with_feature)=",len(data_with_feature),"len(data_without_feature)=",len(data_without_feature),"\n"

		IG = entropy_s - ((float(len(data_with_feature))/len(data))*entropy_s_1) - (float((len(data_without_feature))/len(data))*entropy_s_2)

		return IG

	def find_best_feature(self,data):
		IG_MAX = -1
		best_feature = -1

		for feature in self.features_to_use:
			ig = self.find_IG(feature,data)
			print "ig of",feature,"=",ig
			if(ig>IG_MAX):
				best_feature = feature
				IG_MAX = ig

		print best_feature

		return best_feature

	def split(self,feature,data):
		left = []
		right = []
		string = str(feature)+":"
		for d in data:
			if string in d:
				left.append(d)
			else:
				right.append(d)

		return left,right

	def train_tree(self,data):
		print len(data), len(self.features_to_use)
		if len(data)!=0 and len(self.features_to_use)!=0:	
			best_feature = self.find_best_feature(data)
			left,right = self.split(best_feature,data)
			self.features_to_use.remove(best_feature)
			dict = {}
			dict[len(self.Nodes)] = best_feature 
			self.Nodes.append(dict)
			self.train_tree(left)
			self.train_tree(right)

	def display_tree(self):
		print self.Nodes

def sample_database():
	train_org_file = open("aclImdb_v1/aclImdb/train/labeledBow.feat","r")
	train_new_file = open("train.feat","w")
	test_org_file = open("aclImdb_v1/aclImdb/test/labeledBow.feat","r")
	test_new_file = open("test.feat","w")
	vocabulary_file = open("aclImdb_v1/aclImdb/imdb.vocab","r")

	total_features = 0
	for line in vocabulary_file:
		total_features += 1

	n_pos = 0
	n_neg = 0
	m = 0
	train_data = []
	test_data = []

	for line in train_org_file:
		m += 1
		if random.choice([1,2,3,4,5,6,7,8,9,10,11,12,13,14])==1 and n_neg+n_pos<1000:
			if line[0]>="7" and n_pos<500:
				train_new_file.write(line)
				train_data.append(line)
				n_pos+=1
			if line[0]<="4" and n_neg<500:
				train_new_file.write(line)
				train_data.append(line)
				n_neg+=1


		if n_neg+n_pos>=1000:
			break

	# print n_neg, n_pos
	# print m

	n_pos=0
	n_neg=0
	m=0
	for line in test_org_file:
		m+=1
		if random.choice([1,2,3,4,5,6,7,8,9,10,11,12,13,14])==1 and n_neg+n_pos<1000:
			if line[0]>="7" and n_pos<500:
				test_new_file.write(line)
				n_pos+=1
			if line[0]<="4" and n_neg<500:
				test_new_file.write(line)
				n_neg+=1

		if n_neg+n_pos>=1000:
			break

	# print n_neg, n_pos
	# print m

	train_org_file.close()
	test_org_file.close()

	return train_data, test_data, total_features

if __name__ == '__main__':
	train_data,test_data,total_features = sample_database()


	T = Decision_Tree(total_features) 
	T.train_tree(train_data)
	T.display_tree()

	










