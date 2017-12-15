import os
import json
import datetime

path = "flightdata"
args = {"daily_minimum_only":True, "nonstop":False,"remove_recent":True,"recent":"1210"}
AIRPORTS = {
	'bos': 0, 
	'chi': 1,
	'ord': 1,
	'mdw': 1, 
	'pdx': 2, 
	'lax': 3, 
	'lga': 4
}

def getTOD(hour):
	if hour > 18:
		time_of_day = 3
	elif hour > 12:
		time_of_day = 2
	elif hour > 6:
		time_of_day = 1
	else:
		time_of_day = 0
	return time_of_day

def getLine(date_time_dir_name, d, flight_minimums, flight_key):
	if flight_key not in flight_minimums or flight_minimums[flight_key] >= float(d['price']):
			flight_minimums[flight_key] = float(d['price'])
			label = 1
	else:
		label = 0
	time_of_request = datetime.datetime.strptime(date_time_dir_name, "%m-%d-%Y-%H:%M")
	request_key = date_time_dir_name[0:2]+date_time_dir_name[3:5]+ date_time_dir_name[11:13]
	if args["remove_recent"] and date_time_dir_name[0:4] == args["recent"]:
		return "" # don't include most recent day since those labels will likely be 1 and are not representative
	airport = AIRPORTS[d['arrival'].lower()]
	flight_mon, flight_day, flight_yr = int(flight_key[:2]),int(flight_key[2:4]), 2000+int(flight_key[4:6])
	flight_hr, flight_min = d['time'].split(":")
	flight_hr = int(flight_hr)
	if d['time'][-2] == 'p':
		flight_hr += 12
		flight_hr %= 24
	flight_min = int(flight_min[:-2])
	flight_date = datetime.date(flight_yr, flight_mon, flight_day)
	flight_time = datetime.time(flight_hr, flight_min)
	time_of_flight = datetime.datetime.combine(flight_date, flight_time)
	hours_before = int((time_of_flight - time_of_request).total_seconds()//3600)
	weekday = time_of_request.weekday()
	flight_weekday = flight_date.weekday()
	# times of day: 0 -> 12am-6am, 1 -> 6am-12pm, 2 -> 12pm-6pm, 3 -> 6pm-12am
	time_of_day = getTOD(flight_hr)
	request_tod = getTOD(time_of_request.hour)
	# Format of file lines:
	# request_key format: MMDDHH
	# label: 1 if should buy, 0 otherwise
	# request_key flight_date_key request_weekday flight_weekday request_tod time_of_day hours_before stops duration price airport label
	return "{} {} {} {} {} {} {} {} {} {} {} {}".format(request_key, flight_key[0:4], weekday, flight_weekday, request_tod, time_of_day, hours_before, d['stops'], d['duration'], int(float(d['price'])), airport, label)

def printData(date_time_dir_name, data, flight_minimums, flight_key):
	if args["nonstop"]:
		data = [d for d in data if d['stops'] == 0]
	if len(data) == 0:
		return
	if args["daily_minimum_only"]:
		cheapest = float(data[0]['price']) +1
		best = ""
		for d in data:
			if float(d['price']) < cheapest:
				cheapest = float(d['price'])
				best = getLine(date_time_dir_name, d, flight_minimums, flight_key)
		if len(best) > 0:
			print(best)
	else:
		for d in data:
			print(getLine(data_time_dir_name, d))

if __name__ == '__main__':
	flight_minimums = {}
	for date_time_dir_name in os.listdir(path)[::-1]:
		abs_dir_path_name = "{}/{}".format(path,date_time_dir_name)
		if os.path.isdir(abs_dir_path_name):
			for flight_file_name in os.listdir(abs_dir_path_name):
				if not flight_file_name.startswith('.'):
					abs_file_path_name = "{}/{}".format(abs_dir_path_name, flight_file_name)
					flight_key = flight_file_name[0:2]+flight_file_name[3:5]+flight_file_name[8:10]+flight_file_name[15:18]
					data = json.load(open(abs_file_path_name))
					if len(data) == 0:
						print("removed")
						os.remove(abs_file_path_name)
					else:
						#print(flight_file_name)
						printData(date_time_dir_name, data, flight_minimums, flight_key)

