from os import path
import os
from glob import glob
import sys
from lxml import etree
import xml.etree.ElementTree as ET
import xml.dom.minidom as md

#def suppr_num_page(file_path):
#	with open(file_path, "r") as file :
#		for line in file :
#			if(line.startswith("�") or line.endswith("�",3)):
#				return 0
				#print(line)

def suppr_num_page(file_path):
	with open(file_path, "r+") as file :
		for line in file :
			for char in line :
				if(char=="�"):
					line=line.replace("�",'')##marche pas, enlever les �
					#print(line)		

def stat_file(file_path):
	charac = 0
	words = 0
	phrase = 0
	count = 0
	with open(file_path, "r") as file :
          for line in file :
            charac += len(line)
            word = line.split()
            words += len(word)
            count += line.count("�")
	print('No. of occurrences of - �: ', count)
	print("Nombre de caractères : ",charac) 
	print("Nombre de mots : ",words) 	
	return 0

def main():
	stat_file("fondation.txt")
	#suppr_num_page("fondation.txt")

if __name__ == "__main__":
    main()