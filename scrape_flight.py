#!/usr/bin/env python

import json
import requests
from lxml import html
from collections import OrderedDict
import argparse
from datetime import datetime, timedelta
import os

NUM_DAYS = 21

def parse(source,destination,date):
	url = "https://www.expedia.com/Flights-Search?trip=oneway&leg1=from:{0},to:{1},departure:{2}TANYT&passengers=adults:1,children:0,seniors:0,infantinlap:Y&options=cabinclass%3Aeconomy&mode=search&origref=www.expedia.com".format(source,destination,date, 
		headers={'User-Agent': 'Flight data by qiqiren'})
	response = requests.get(url)
	parser = html.fromstring(response.text)
	json_data_xpath = parser.xpath("//script[@id='cachedResultsJson']//text()")
	try:
		raw_json =json.loads(json_data_xpath[0])
	except:
		print("error")
		return "error"
	flight_data = json.loads(raw_json["content"])
	# print(flight_data['legs'])
	lists=[]

	for i in flight_data['legs'].keys():
		# print(flight_data['legs'][i])
		total_distance =  flight_data['legs'][i]['formattedDistance']
		exact_price = flight_data['legs'][i]['price']['totalPriceAsDecimal']

		departure_airport_code = flight_data['legs'][i]['departureLocation']['airportCode']
		
		arrival_airport_code = flight_data['legs'][i]['arrivalLocation']['airportCode']

		airline_code = flight_data['legs'][i]['carrierSummary']['airlineCodes'][0]
		
		stops = flight_data['legs'][i]["stops"]
		flight_hour = flight_data['legs'][i]['duration']['hours']
		flight_minutes = flight_data['legs'][i]['duration']['minutes']
		duration = flight_hour*60 + flight_minutes

		formatted_price = "{0:.2f}".format(exact_price)
		
		date = flight_data['legs'][i]['departureTime']['date']
		time = flight_data['legs'][i]['departureTime']['time']
		flight_info={'stops':stops,
			'price':formatted_price,
			'departure':departure_airport_code,
			'arrival':arrival_airport_code,
			'duration': duration,
			'airline':airline_code,
			'date': date,
			'time': time
		}
		lists.append(flight_info)
	sortedlist = sorted(lists, key=lambda k: k['price'],reverse=False)
	return sortedlist

if __name__=="__main__":
	now = datetime.now()
	date_now = now.strftime("%m-%d-%Y")
	time_now = now.strftime("%H:%M")
	sources = ["sfo"]
	destinations = ['bos', 'chi', 'pdx', 'lax', 'lga']
	dates = [(now + timedelta(days=i)).strftime("%m-%d-%Y") for i in range(NUM_DAYS)]

	directory = 'flightdata/{}-{}'.format(date_now,time_now)
	if not os.path.exists(directory):
		os.mkdir(directory)
	for source in sources:
		for destination in destinations:
			for date in dates:
				print("{}, {}, {}".format(source,destination,date))
				scraped_data = parse(source,destination,date)
				# print("Writing data to output file")
				if scraped_data != "error":
					with open('{}/{}-{}-{}-flight-results.json'.format(directory,date,source,destination),'w') as fp:
						json.dump(scraped_data,fp,indent = 4)
