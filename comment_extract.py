import lxml         #It provides safe and convenient access to some libraries like libxml2 and libxslt.
import requests     # Licensed HTTP library, written in Python used by humans to interact with the language.
import time         #function returns the number of seconds passed since epoch.
import sys          #module provides information about constants.
import progress_bar as PB  #A simple way of providing an informative and clean progress bar on the terminal.
import googleapiclient.discovery


#take Youyube Links
YOUTUBE_IN_LINK = 'https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&maxResults=100&order=relevance&pageToken={pageToken}&videoId={videoId}&key={key}'
YOUTUBE_LINK = 'https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&maxResults=100&order=relevance&videoId={videoId}&key={key}'

api_service_name = "youtube"
api_version = "v3"
key = "AIzaSyAyKJfBn1ysopYBk-n3mXd2TZ5mSWkegwo"

youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey = key)


#Extract comments using link and number	(Key).
def commentExtract(videoId, count = -1):
	print ("Comments downloading")
	#request = youtube.commentThreads().list(part="snippet",	maxResults=100,	videoId=videoId)
	
	#response = request.execute()

	headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.63 Safari/537.36'}
	page_info = requests.get(YOUTUBE_LINK.format(videoId = videoId, key = key), headers)
	#page_info = response
	while page_info.status_code != 200:
		if page_info.status_code != 429:
			print ("Comments disabled")
			sys.exit()

		time.sleep(20)
		page_info = requests.get(YOUTUBE_LINK.format(videoId = videoId, key = key))

	page_info = page_info.json();#print(page_info);
    
	comments = []
	co = 0;
	for i in range(len(page_info['items'])):
		comments.append(page_info['items'][i]['snippet']['topLevelComment']['snippet']['textOriginal'])
		co += 1
		if co == count:
			PB.progress(co, count, cond = True)
			print ()
			return comments

	PB.progress(co, count)
	# INFINTE SCROLLING
	while 'nextPageToken' in page_info:
		temp = page_info
		page_info = requests.get(YOUTUBE_IN_LINK.format(videoId = videoId, key = key, pageToken = page_info['nextPageToken']))

		while page_info.status_code != 200:
			time.sleep(20)
			page_info = requests.get(YOUTUBE_IN_LINK.format(videoId = videoId, key = key, pageToken = temp['nextPageToken']))
		page_info = page_info.json()

		for i in range(len(page_info['items'])):
			comments.append(page_info['items'][i]['snippet']['topLevelComment']['snippet']['textOriginal'])
			co += 1
			if co == count:
				PB.progress(co, count, cond = True)
				print ()
				return comments
		PB.progress(co, count)
	PB.progress(count, count, cond = True);print(comments);
	return comments

#print(commentExtract("16x08VVantY",10))