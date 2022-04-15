# import pymongo
from pymongo import MongoClient


conn_str = "mongodb+srv://kimia:EaoHrgfNcva27lMT@cluster0.6ihoy.mongodb.net/match-it?retryWrites=true&w=majority"
# conn_str = "mongodb+srv://<username>:<password>@<cluster-address>/test?retryWrites=true&w=majority"

client = MongoClient(conn_str)

collection_stylist = client.get_database("match-it").get_collection("stylist")
collection_style_seeker = client.get_database("match-it").get_collection("styleseeker")

documents_stylist = collection_stylist.find({})
documents_style_seeker = collection_style_seeker.find({})

# for i in range(1000):
    
for document in documents_stylist:
    print(document['interests'])
    
for document in documents_style_seeker:
    print(document['interests'])
    
    
## interests
## ["18-29 group", "$15-$20", "funky", ""]