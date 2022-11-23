import csv
import re
import os
import glob
from shutil import copyfile
from datetime import datetime
from bs4 import BeautifulSoup 
import datetime


def read_file_list(indir, d):
	opt_notes = []
	with open(indir, 'r') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=d)
		for row in spamreader:
			opt_notes += [row]
	return opt_notes


def read_txt(indir):
	f = open(indir,'r')
	txt = f.read()
	f.close()
	return txt


def cam_cp(cp):
	cp = cp.strip()
	cam = ''
	if cp in ['encephalopathy','agit','ams','fluctu', 'confusion']:
		cam = 'cam_a'
	elif cp in ['inattention']:
		cam = 'cam_b'
	elif cp in ['hallucination', 'disorganized_thinking', 'disorient']:
		cam = 'cam_c'
	elif cp in ['disconnected']:
		cam = 'cam_d'
	elif cp == 'delirium':
		cam = 'delirium'
	return cam

def cam_original(cam_list):
	cam_list = [i.lower() for i in cam_list if i != '']
	delirium_status = 0
	if 'cam_a' in cam_list and 'cam_b' in cam_list and 'cam_d' in cam_list and 'cam_c' in cam_list:
		delirium_status = 1
	elif 'cam_a' in cam_list and 'cam_b' in cam_list and 'cam_d' in cam_list:
		delirium_status = 1
	elif 'cam_a' in cam_list and 'cam_b' in cam_list and 'cam_c' in cam_list:
		delirium_status = 1
	elif 'delirium' in cam_list or 'deli_cam' in cam_list:
		delirium_status = 1
	elif cam_list != []:
		delirium_status = 0
	return delirium_status	


def output_evidence():
	nlp_patient_norm,nlp_patient_cam = {},{}
	nlp_result_txt = glob.glob('/AgingNLP/delirium/output/*.ann')
	print('loading finished')
	
	for i in nlp_result_txt:
		note = read_file_list(i, '\t')
		if note == []:
			continue	
		for row in note:
			certainty = row[6]
			status = row[7]
			experiencer = row[8]
			if 'Positive' in certainty and 'Patient' in experiencer:
				norm = row[9].lower().split('\"')[1]
				cam = cam_cp(norm)
				file_name = row[0]
				if file_name not in nlp_patient_norm:
					nlp_patient_norm[file_name] = [norm]
				else:
					nlp_patient_norm[file_name] = nlp_patient_norm[file_name]+[norm]	
				if file_name not in nlp_patient_cam:
					nlp_patient_cam[file_name] = [cam]
				else:
					nlp_patient_cam[file_name] = nlp_patient_cam[file_name]+[cam]						

	with open('summarized_result.csv', 'w') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter='|')
		for file_name in nlp_patient_cam:
			nlpcam = cam_original(nlp_patient_cam[file_name])
			spamwriter.writerow([file_name, nlp_patient_norm[file_name], nlp_patient_cam[file_name], nlpcam])


				
		
output_evidence()