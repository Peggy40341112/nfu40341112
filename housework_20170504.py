from instagram.client import InstagramAPI
access_token = "1532802561.870ffda.2cb6124d70d242c5a7ec54264bfb6688"
client_secret = "348c5332c8134906a99b951421eb30b6"

api = InstagramAPI(access_token=access_token, client_secret=client_secret,)
user_id = api.user_search('dessert1112')[0].id
recent_media, next_ = api.user_recent_media(user_id=user_id, count=5)

for media in recent_media:
   print (media.caption.text)
   print ('<img src="%s"/>' % media.images['thumbnail'].url)

