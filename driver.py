import comment_extract as CE
import sentimentYouTube as SYT

def main():
	# EXAMPLE VideoId = 'tCXGJQYZ9JA'
    videoLink='https://www.youtube.com/watch?v=MY5SatbZMAo'
    videoId = videoLink.split("=", 1)[1]
    #videoId = input("Enter VideoId : ")
	# Fetch the number of comments
	# if count = -1, fetch all comments
    #count = int(input("Enter no. of comments to extract : "))
    comments = CE.commentExtract(videoId, 10)
    #print(comments)
    SYT.sentiment(comments,videoId)
   
if __name__ == '__main__':
	main()
